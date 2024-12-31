import torch as th
from torch import nn
from typing import List, Tuple
from stable_baselines3.common.distributions import kl_divergence
from sb3_contrib.trpo.trpo import TRPO


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

  def learn(self, **params):
    params["tb_log_name"] = self.tb_log_name
    return super().learn(**params)


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

      search_direction = self._solve_conjugate_gradient(
          hessian_vector_product_fn, policy_objective_gradients
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
    explained_var = self._get_explained_variance()
    self.logger.record("train/policy_objective", float(th.mean(th.tensor(policy_objective_values))))
    self.logger.record("train/value_loss", float(th.mean(th.tensor(value_losses))))
    self.logger.record("train/kl_divergence_loss", float(th.mean(th.tensor(kl_divergences))))
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/is_line_search_success", float(th.mean(th.tensor(line_search_results, dtype=th.float32))))
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

  def _solve_conjugate_gradient(self, f_Ax, b):
    x = th.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = th.dot(r, r)
    for _ in range(self.cg_max_steps):
      Ap = f_Ax(p)
      alpha = rr / (th.dot(p, Ap) + 1e-8)
      x += alpha * p
      r -= alpha * Ap
      rr_new = th.dot(r, r)
      if rr_new < 1e-10:
        break
      beta = rr_new / rr
      p = r + beta * p
      rr = rr_new
    return x

  def _get_explained_variance(self):
    # Convert returns and values to PyTorch tensors
    returns = th.tensor(self.rollout_buffer.returns, dtype=th.float32, device=self.device)
    values = th.tensor(self.rollout_buffer.values, dtype=th.float32, device=self.device)

    # Calculate explained variance
    explained_var = 1 - th.var(returns - values) / th.var(returns)
    return explained_var
