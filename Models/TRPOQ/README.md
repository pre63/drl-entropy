# TRPOQ: Trust Region Policy Optimization with Quantile-Based Value Estimation

**TRPOQ** is a reinforcement learning algorithm that extends the standard TRPO (Trust Region Policy Optimization) by incorporating a quantile-based critic architecture. This modification aims to reduce overestimation bias and stabilize advantage estimation during policy updates, inspired by concepts from distributional reinforcement learning and quantile regression.

TRPOQ introduces multiple critics with quantile regression and uses a conservative truncation strategy to improve the stability of value estimation and advantage calculation, ultimately leading to more reliable policy updates in continuous control tasks.


## Key Features and Innovations

### 1. **Quantile-Based Value Function Estimation**
   - The standard TRPO critic is replaced with an ensemble of quantile critics.
   - Each critic outputs multiple quantiles representing a distribution over returns instead of a single point estimate.
   - The critics use quantile regression to approximate the distribution of returns more effectively.

### 2. **Truncation of Quantiles for Advantage Estimation**
   - When calculating the advantage function, only the lower portion of the quantile estimates is retained.
   - The highest quantile values are discarded to prevent overestimation bias.
   - This conservative approach ensures more stable advantage estimates, reducing variance in policy updates.

### 3. **Ensemble of Value Networks**
   - Multiple value networks (critics) are maintained, each estimating a separate return distribution.
   - The ensemble improves stability by reducing the variance and bias of value estimation through redundancy and averaging.

### 4. **Distributional Advantage Calculation**
   - Advantage calculation is modified to use the conservative truncated quantile estimates rather than a single point estimate.
   - The advantage is computed as:

   $A(s, a) = r(s, a) + \gamma \bar{V}_{\text{trunc}}(s') - \bar{V}_{\text{trunc}}(s)$]

   where \( \bar{V}_{\text{trunc}}(s) \) is the truncated mean of the lower quantiles.

### 5. **Trust Region Optimization (TRPO) Backbone**
   - TRPOQ retains the KL-divergence constraint optimization from standard TRPO.
   - A line search is used to enforce a constraint on the KL divergence between the old and new policy, ensuring stable updates.


## Algorithm Steps (TRPOQ)

1. **Collect Trajectories:** Sample trajectories using the current policy.
2. **Quantile Value Estimation:** Compute value estimates using the ensemble of quantile-based critics.
3. **Truncate Quantile Estimates:** Discard the top quantiles and compute the conservative truncated mean for value estimation.
4. **Advantage Calculation:** Compute the advantage using the conservative truncated value estimates.
5. **Policy Update (TRPO):**
   - Compute the policy gradient using the advantage estimates.
   - Perform a constrained optimization step using a KL-divergence trust region.
6. **Value Function Update:**
   - Update the quantile critics using a quantile regression loss (Huber loss).
7. **Repeat:** Continue the process for multiple episodes until convergence.


## Mathematical Formulation

### **Policy Objective (with TRPO Constraint)**

$
L(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_{\text{trunc}}(s_t, a_t) \right]
$

subject to the KL divergence constraint:

$
D_{KL}(\pi_\theta || \pi_{\theta_\text{old}}) \leq \delta
$

### **Quantile-Based Value Estimation**
The value function estimates multiple quantiles:

$
Q_\theta(s, a) = \{ q_\theta^1, q_\theta^2, \dots, q_\theta^K \}
$

The truncated value estimate for advantage computation:

$
\bar{V}_{\text{trunc}}(s) = \frac{1}{K_{\text{used}}} \sum_{i=1}^{K_{\text{used}}} q_\theta^i
$

where \( K_{\text{used}} \) is the number of quantiles retained after truncation.


## Advantages of TRPOQ

- **Reduced Overestimation Bias:** By discarding the highest quantiles, TRPOQ mitigates overestimation common in Q-value based methods.
- **Improved Stability:** Using multiple critics and quantile regression stabilizes the advantage estimates.
- **Better Sample Efficiency:** The conservative value estimation approach allows more reliable policy updates.
- **KL-Constrained Optimization:** Retains the stable KL-constrained policy update mechanism of TRPO.


## Differences from Standard TRPO and QR-DQN

| Feature                              | TRPOQ                              | TRPO                          | QR-DQN                       |
|-------------------------------------|------------------------------------|-------------------------------|------------------------------|
| **Algorithm Type**                  | Policy Gradient (On-Policy)       | Policy Gradient (On-Policy)  | Value-Based (Off-Policy)    |
| **Critic Type**                     | Quantile-Based Critics Ensemble   | Single Value Function        | Quantile Regression Q-Value |
| **Quantile Estimation**             | Yes, with Truncation              | No                           | Yes, Full Distribution      |
| **Overestimation Handling**         | Conservative Truncated Quantiles  | No Handling                  | Full Quantile Estimation    |
| **Policy Update Mechanism**         | KL-Constrained Optimization       | KL-Constrained Optimization  | Temporal Difference (TD)    |
| **Data Efficiency**                 | Lower (On-Policy)                 | Lower (On-Policy)            | Higher (Off-Policy)         |


## Use Cases
- **Continuous Control Tasks:** Suitable for tasks like robotic control and Mujoco environments where continuous actions require stable policy updates.
- **High Variance Environments:** Effective when advantage estimates suffer from high variance due to reward noise.
- **Risk-Averse Training:** The conservative advantage estimation makes TRPOQ useful in safety-critical environments where overoptimistic policies can be dangerous.


## Future Extensions and Improvements
- **Multi-Step Quantile Regression:** Incorporating multi-step return estimation could further improve sample efficiency.
- **Risk-Aware Exploration:** Adjusting the truncation level dynamically based on uncertainty could balance exploration and exploitation better.
- **Combination with PPO:** Adapting the clipped surrogate objective from PPO with quantile estimation could simplify policy updates while keeping stability.


## Conclusion
TRPOQ introduces a novel way to stabilize policy optimization by integrating quantile regression and conservative advantage estimation into the TRPO framework. By leveraging an ensemble of critics with truncated quantile estimates, it reduces overestimation bias and enhances training stability, making it suitable for complex continuous control tasks.
