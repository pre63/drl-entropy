import torch as th
from torch import nn

from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance

from sb3_contrib.common.utils import conjugate_gradient_solver
from Models.TRPO import TRPO


class EnTRPOR(TRPO):
  """
    EnTRPOR: Entropy-Regularized Trust Region Policy Optimization with Reinforcement Learning

    Like TRPOR but with entropy regularization for line search.

    This is an extension of the standard Trust Region Policy Optimization (TRPO) algorithm
    that incorporates entropy regularization into the policy objective. The entropy bonus
    encourages exploration and prevents premature convergence to deterministic policies.

    Key Features:
    - Adds an entropy bonus term to the policy objective to promote exploration.
    - Retains the KL-divergence constraint from TRPO for stable policy updates.
    - The entropy coefficient (`ent_coef`) is tunable for balancing exploration and exploitation.
    - Suitable for environments with sparse rewards where additional exploration is needed.

    Mathematical Formulation:
    -------------------------
    Standard TRPO objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) ]

    EnTRPOR modified objective:
        L(θ) = E_t [ (π_θ(a_t | s_t) / π_θ_old(a_t | s_t)) * Â(s_t, a_t) + α * H(π_θ) ]

    where:
    - π_θ is the current policy.
    - π_θ_old is the old policy.
    - Â is the advantage function.
    - α (`ent_coef`) is the entropy coefficient.
    - H(π_θ) is the entropy of the policy.

    Parameters:
    -----------
    policy : Union[str, type[ActorCriticPolicy]]
        The policy model to be used (e.g., "MlpPolicy").
    env : Union[GymEnv, str]
        The environment to learn from.
    ent_coef : float, optional
        Entropy coefficient controlling the strength of the entropy bonus (default: 0.01).
    learning_rate : Union[float, Schedule], optional
        Learning rate for the optimizer (default: 1e-3).
    n_steps : int, optional
        Number of steps to run per update (default: 2048).
    batch_size : int, optional
        Minibatch size for the value function updates (default: 128).
    gamma : float, optional
        Discount factor for the reward (default: 0.99).
    cg_max_steps : int, optional
        Maximum steps for the conjugate gradient solver (default: 10).
    target_kl : float, optional
        Target KL divergence for policy updates (default: 0.01).

    Methods:
    --------
    learn(**params)
        Trains the agent with the specified parameters, logging results under `"EnTRPOR"`.

    train()
        Performs a single policy update using the KL-constrained optimization approach
        with entropy regularization.

    Differences from Standard TRPO:
    -------------------------------
    - **Entropy Bonus:** Adds entropy to the policy objective for better exploration.
    - **Policy Objective:** Modified to include the entropy coefficient (`ent_coef`).
    - **Line Search:** Considers the entropy term while checking policy improvement.
    - **Logging:** Logs entropy-regularized objectives and KL divergence values.

  """

  def __init__(
      self,
      *args,
      ent_coef: float = 0.01,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.ent_coef = ent_coef
    self.tb_log_name = "EnTRPOR"

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
      if self.sub_sampling_factor > 1:
        rollout_data = type(rollout_data)(
            rollout_data.observations[:: self.sub_sampling_factor],
            rollout_data.actions[:: self.sub_sampling_factor],
            None,
            rollout_data.old_log_prob[:: self.sub_sampling_factor],
            rollout_data.advantages[:: self.sub_sampling_factor],
            None,
        )

      actions = rollout_data.actions
      if self.action_space.__class__.__name__ == "Discrete":
        actions = rollout_data.actions.long().flatten()

      with th.no_grad():
        old_distribution = self.policy.get_distribution(rollout_data.observations)

      distribution = self.policy.get_distribution(rollout_data.observations)
      log_prob = distribution.log_prob(actions)
      ent = distribution.entropy().mean()

      advantages = rollout_data.advantages
      if self.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      ratio = th.exp(log_prob - rollout_data.old_log_prob)
      policy_objective = (advantages * ratio).mean() + self.ent_coef * ent
      kl_div = kl_divergence(distribution, old_distribution).mean()

      self.policy.optimizer.zero_grad()

      actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
          kl_div, policy_objective
      )

      def hessian_vector_product_fn(vec, rg=True): return self.hessian_vector_product(
          actor_params, grad_kl, vec, retain_graph=rg
      )

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
          new_ent = distribution.entropy().mean()
          new_policy_objective = (advantages * ratio).mean() + self.ent_coef * new_ent
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
        for param in actor_params:
          param.grad = None
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


def sample_entrpor_params(trial, n_actions, n_envs, additional_args):
  # Define the EnTRPOR-specific optimization space
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
      "n_envs": n_envs,
      "n_actions": n_actions,
      **additional_args,
  }
  return params
