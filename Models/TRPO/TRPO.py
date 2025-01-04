import warnings
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from sbx.common.on_policy_algorithm import OnPolicyAlgorithmJax
from Models.TRPO.Policies import TRPOPolicy

TRPOSelf = TypeVar("TRPOSelf", bound="TRPO")

class TRPO(OnPolicyAlgorithmJax):
    """
    Trust Region Policy Optimization (TRPO) algorithm.

    Paper: https://arxiv.org/abs/1502.05477

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate (not typically used in TRPO but kept for compatibility)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss (TRPO typically uses one epoch)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param target_kl: Maximum KL divergence between old and new policy
    :param cg_max_steps: Number of iterations for the conjugate gradient algorithm
    :param cg_damping: Damping factor for the conjugate gradient algorithm
    :param backtrack_iters: Number of backtracking iterations for line search
    :param backtrack_coef: Step size reduction factor for line search
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    
    policy_aliases: ClassVar[Dict[str, Type[TRPOPolicy]]] = {
        "MlpPolicy": TRPOPolicy,
        # Add other policy aliases if needed
    }
    policy: TRPOPolicy  # type: ignore[assignment]

    def __init__(
        self,
        policy: Union[str, Type[TRPOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,  # Typically unused in TRPO
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 1,  # TRPO typically uses one epoch
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        target_kl: float = 0.01,
        cg_max_steps: int = 10,
        cg_damping: float = 0.1,
        backtrack_iters: int = 10,
        backtrack_coef: float = 0.8,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
        n_critic_updates: int = 5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        self.backtrack_iters = backtrack_iters
        self.backtrack_coef = backtrack_coef
        self.n_critic_updates = n_critic_updates

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )

            self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor
            self.vf = self.policy.vf

    def conjugate_gradient(self, Avp_func, b, cg_max_steps, residual_tol=1e-10):
        """
        Conjugate Gradient solver to solve Ax = b.
        """
        x = jnp.zeros_like(b)
        r = b
        p = r
        rsold = jnp.dot(r, r)

        def cond(val):
            i, rsold, r, p, x = val
            return jnp.logical_and(i < cg_max_steps, rsold > residual_tol)

        def body(val):
            i, rsold, r, p, x = val
            Ap = Avp_func(p)
            alpha = rsold / jnp.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = jnp.dot(r, r)
            beta = rsnew / rsold
            p = r + beta * p
            return (i + 1, rsnew, r, p, x)

        i, rsold, r, p, x = jax.lax.while_loop(cond, body, (0, rsold, r, p, x))
        return x

    def fisher_vector_product(self, params, observations, old_log_prob, actions):
        """
        Compute the Fisher-vector product.
        """
        def kl_div(params):
            dist = self.policy.actor.apply_fn(params, observations)
            log_prob = dist.log_prob(actions)
            ratio = jnp.exp(log_prob - old_log_prob)
            return jnp.mean((ratio - 1.0) ** 2)

        grad_kl = jax.grad(kl_div)(params)
        def hvp(v):
            return jax.grad(lambda p: jnp.dot(jax.grad(kl_div)(p), v))(params)
        return hvp(grad_kl) + self.cg_damping * v

    def _compute_loss_and_grad(self, params, vf_params, observations, actions, advantages, returns, old_log_prob):
        """
        Compute the surrogate loss and gradients for the actor and critic.
        """
        def surrogate_loss(params):
            dist = self.policy.actor.apply_fn(params, observations)
            log_prob = dist.log_prob(actions)
            ratio = jnp.exp(log_prob - old_log_prob)
            return jnp.mean(ratio * advantages)

        pg_loss, pg_grad = jax.value_and_grad(surrogate_loss)(params)
        
        def value_loss_fn(vf_params):
            vf_values = self.policy.vf.apply_fn(vf_params, observations).flatten()
            return jnp.mean((returns - vf_values) ** 2)

        value_loss, vf_grad = jax.value_and_grad(value_loss_fn)(vf_params)
        return pg_loss, pg_grad, value_loss, vf_grad

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Compute current clip range if needed (TRPO doesn't typically use clip_range)
        # clip_range = self.clip_range_schedule(self._current_progress_remaining)
        # Critic update


        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.flatten().astype(np.int32)
                else:
                    actions = rollout_data.actions

                observations = rollout_data.observations
                advantages = rollout_data.advantages
                returns = rollout_data.returns
                old_log_prob = rollout_data.old_log_prob

                # Convert to JAX arrays
                observations = jnp.array(observations)
                actions = jnp.array(actions)
                advantages = jnp.array(advantages)
                returns = jnp.array(returns)
                old_log_prob = jnp.array(old_log_prob)

                # Compute loss and gradients
                pg_loss, pg_grad, value_loss, vf_grad = self._compute_loss_and_grad(
                    self.policy.actor_state.params,
                    self.policy.vf_state.params,
                    observations,
                    actions,
                    advantages,
                    returns,
                    old_log_prob
                )

                # Define the Avp function for the conjugate gradient
                def Avp(v):
                    return self.fisher_vector_product(
                        self.policy.actor_state.params,
                        observations,
                        old_log_prob,
                        actions
                    ) * v

                # Compute the policy gradient step direction
                step_dir = self.conjugate_gradient(Avp, pg_grad, self.cg_max_steps)

                # Compute the step size
                shs = 0.5 * jnp.dot(step_dir, Avp(step_dir))
                lm = jnp.sqrt(shs / self.target_kl)
                full_step = step_dir / lm

                # Line search to ensure KL constraint
                def policy_update(params, step):
                    return self.policy.actor.apply_fn(params + step, observations)

                params = self.policy.actor_state.params
                new_params = params + full_step

                # Optional: Implement a more sophisticated line search
                # For simplicity, we apply the full step here
                self.policy.actor_state = self.policy.actor_state.replace(params=new_params)

                # Update the value function
                new_vf_params = self.policy.vf_state.params - self.learning_rate * vf_grad  # TRPO may use different update
                self.policy.vf_state = self.policy.vf_state.replace(params=new_vf_params)

                # Logging
                self.logger.record("train/pg_loss", pg_loss.item())
                self.logger.record("train/value_loss", value_loss.item())
                self.logger.record("train/target_kl", self.target_kl)
        
        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: TRPOSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> TRPOSelf:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
  