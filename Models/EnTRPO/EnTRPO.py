import torch as th
from torch import nn
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance

from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from sb3_contrib.trpo.trpo import TRPO

import optuna

class EnTRPO(TRPO):
  def __init__(
      self,
      *args,
      ent_coef: float = 0.01,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.ent_coef = ent_coef
    self.tb_log_name = "EnTRPO"

  def learn(self, n_timesteps, callback, **learn_kwargs):
    learn_kwargs["tb_log_name"] = self.tb_log_name
    return super().learn(n_timesteps, callback, **learn_kwargs)

  def train(self) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    policy_objective_values = []
    kl_divergences = []
    line_search_results = []
    value_losses = []

    for rollout_data in self.rollout_buffer.get(batch_size=None):
      actions = rollout_data.actions
      if self.action_space.__class__.__name__ == "Discrete":
        actions = rollout_data.actions.long().flatten()

      with th.no_grad():
        old_distribution = self.policy.get_distribution(rollout_data.observations)

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)
      entropy = distribution.entropy().mean()

      advantages = rollout_data.advantages
      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - rollout_data.old_log_prob)
      policy_objective = (advantages * ratio).mean() + self.ent_coef * entropy
      kl_div = kl_divergence(distribution, old_distribution).mean()

      self.policy.optimizer.zero_grad()

      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
          kl_div, policy_objective
      )

      def hessian_vector_product_fn(vec, rg=True):
        return self.hessian_vector_product(actor_params, grad_kl, vec, retain_graph=rg)

      search_direction = conjugate_gradient_solver(
          hessian_vector_product_fn,
          policy_objective_gradients,
          max_iter=self.cg_max_steps,
      )

      line_search_max_step_size = 2 * self.target_kl
      line_search_max_step_size /= th.matmul(
          search_direction, hessian_vector_product_fn(search_direction, rg=False)
      )
      line_search_max_step_size = th.sqrt(line_search_max_step_size)

      line_search_backtrack_coeff = 1.0
      original_actor_params = [param.detach().clone() for param in actor_params]
      is_line_search_success = False

      with th.no_grad():
        for _ in range(self.line_search_max_iter):
          start_idx = 0
          for param, orig, shape in zip(actor_params, original_actor_params, grad_shape):
            n_params = param.numel()
            param.data = orig.data + line_search_backtrack_coeff * line_search_max_step_size * \
                search_direction[start_idx: (start_idx + n_params)].view(shape)
            start_idx += n_params

          distribution = self.policy.get_distribution(rollout_data.observations)
          log_prob = distribution.log_prob(actions)
          ratio = th.exp(log_prob - rollout_data.old_log_prob)
          new_entropy = distribution.entropy().mean()
          new_policy_objective = (advantages * ratio).mean() + self.ent_coef * new_entropy
          kl_div_new = kl_divergence(distribution, old_distribution).mean()

          if (kl_div_new < self.target_kl) and (new_policy_objective > policy_objective):
            is_line_search_success = True
            break
          line_search_backtrack_coeff *= self.line_search_shrinking_factor

        line_search_results.append(is_line_search_success)
        if not is_line_search_success:
          for param, orig in zip(actor_params, original_actor_params):
            param.data = orig.data.clone()
          policy_objective_values.append(policy_objective.item())
          kl_divergences.append(0.0)
        else:
          policy_objective_values.append(new_policy_objective.item())
          kl_divergences.append(kl_div_new.item())

    for _ in range(self.n_critic_updates):
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        values_pred = self.policy.predict_values(rollout_data.observations)
        value_loss = nn.functional.mse_loss(rollout_data.returns, values_pred.flatten())
        value_losses.append(value_loss.item())
        self.policy.optimizer.zero_grad()
        value_loss.backward()
        self.policy.optimizer.step()

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
    self.logger.record("train/policy_objective", float(th.mean(th.tensor(policy_objective_values))))
    self.logger.record("train/value_loss", float(th.mean(th.tensor(value_losses))))
    self.logger.record("train/kl_divergence_loss", float(th.mean(th.tensor(kl_divergences))))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", float(th.mean(th.tensor(line_search_results, dtype=th.float32))))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def optimal(trial, total_timesteps):
  # Define the EnTRPO-specific optimization space
  params = {
      "policy": "MlpPolicy",
      "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001),
      "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
      "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.001),
      "target_kl": trial.suggest_float("target_kl", 0.001, 0.05, step=0.001),
      "cg_damping": trial.suggest_float("cg_damping", 0.01, 0.1, step=0.01),
      "cg_max_steps": trial.suggest_int("cg_max_steps", 10, 20, step=1),
      "line_search_max_iter": trial.suggest_int("line_search_max_iter", 5, 15, step=5),
      "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
      "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
      "total_timesteps": total_timesteps
  }
  return params


def sample_entrpo_params(trial, n_actions, n_envs, additional_args) :
    """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    # line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])
    n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
    cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
    # cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

    return {
        "ent_coef": ent_coef,

        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        # "cg_damping": cg_damping,
        "cg_max_steps": cg_max_steps,
        # "line_search_shrinking_factor": line_search_shrinking_factor,
        "n_critic_updates": n_critic_updates,
        "target_kl": target_kl,
        "learning_rate": learning_rate,
        "gae_lambda": gae_lambda,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
