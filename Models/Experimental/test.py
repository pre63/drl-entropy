import matplotlib.pyplot as plt
import numpy as np


def test(entropy_values, entropy_coeff_values):
  # Updated parameters for the plot
  # entropy_values = np.linspace(-10, 10, 1000)  # Entropy range
  # entropy_coeff_values = np.linspace(-1, 1, 10)  # Coefficients to test
  min_samples = 0
  max_samples = 10000

  # Plotting
  plt.figure(figsize=(10, 6))
  for coeff in entropy_coeff_values:
    if coeff == 0:  # Skip 0 to avoid division by zero
      continue
    samples = [sampling_strategy(e, coeff, min_samples, max_samples) for e in entropy_values]
    plt.plot(entropy_values, samples, label=f"Entropy Coeff: {coeff:.2f}")

  plt.title("Gradual Linear Sampling Distribution (Max/Min at Zero Entropy)")
  plt.xlabel("Entropy")
  plt.ylabel("Number of Samples")
  plt.legend()
  plt.grid()
  plt.show()


def sampling_strategy(entropy, entropy_coeff, min_samples=0, max_samples=10000):
  """
    Computes the number of samples to draw based on entropy and binary coefficient behavior,
    ensuring a more gradual slope for linear adjustment for both positive and negative coefficients.

    Args:
        entropy: Scalar or array representing the entropy of observations (unnormalized).
        entropy_coeff: Scalar in range [-1, 1], non-zero, that determines max/min behavior.
        min_samples: Minimum number of samples.
        max_samples: Maximum number of samples.

    Returns:
        Number of samples to draw.
    """
  samples_range = max_samples - min_samples

  if entropy_coeff > 0:
    # Gradual scaling with max samples at zero entropy for positive coefficients
    factor = 1 - np.abs(entropy * (1 / (abs(entropy_coeff) * 10)))  # Gradual slope adjustment
    samples = samples_range * factor
  else:
    # Gradual scaling with min samples at zero entropy for negative coefficients
    factor = np.abs(entropy * (1 / (abs(entropy_coeff) * 10)))  # Gradual slope adjustment
    samples = samples_range * factor

  # Ensure samples stay within the min and max bounds
  return np.clip(samples + min_samples, min_samples, max_samples)


def sampling_strategy(entropy, sampling_coef, min_samples=0, max_samples=10000):
  """
    Computes the number of samples to draw based on normalized entropy and a sampling coefficient,
    using linear interpolation (an affine mapping) between the minimum and maximum sample bounds.
    With a positive sampling_coef the interpolation is direct (min at 0 entropy, max at 1),
    whereas with a negative sampling_coef the interpolation is inverted (max at 0 entropy, min at 1).
    This formulation follows from the linear interpolation principle.

    The computation is defined by:
      factor = 0.5 + sampling_coef * (entropy - 0.5)
      samples = min_samples + (max_samples - min_samples) * factor

    Args:
        entropy: A scalar or array representing normalized entropy in [0, 1].
        sampling_coef: A scalar in the range [-1, 1] that guides the behavior. A positive value means
                       sampling increases with entropy, while a negative value means it decreases.
        min_samples: The minimum number of samples.
        max_samples: The maximum number of samples.

    Returns:
        An integer number of samples, clipped to lie between min_samples and max_samples.
    """
  samples_range = max_samples - min_samples
  factor = 0.5 + sampling_coef * (entropy - 0.5)
  samples = min_samples + samples_range * factor
  return int(np.clip(samples, min_samples, max_samples))


def sampling_strategy(entropy, sampling_coef, min_samples=0, max_samples=10000):
  """
    Computes the number of samples to draw by mapping normalized entropy through a smoothstep (cubic Hermite) function
    to achieve a more gradual transition. A positive sampling_coef yields a mapping where maximum samples occur at high entropy,
    whereas a negative sampling_coef inverts the mapping so that maximum samples occur at low entropy.
    The transformation is defined by:
      smooth_entropy = 3 * entropy**2 - 2 * entropy**3
      factor = 0.5 + sampling_coef * (smooth_entropy - 0.5)
      samples = min_samples + (max_samples - min_samples) * factor

    Args:
        entropy: Scalar or array-like normalized entropy in [0, 1].
        sampling_coef: Scalar in [-1, 1] that guides the behavior.
        min_samples: Minimum number of samples.
        max_samples: Maximum number of samples.

    Returns:
        An integer number of samples within [min_samples, max_samples].
    """
  samples_range = max_samples - min_samples
  smooth_entropy = 3 * np.power(entropy, 2) - 2 * np.power(entropy, 3)
  factor = 0.5 + sampling_coef * (smooth_entropy - 0.5)
  samples = min_samples + samples_range * factor
  return int(np.clip(samples, min_samples, max_samples))


def sampling_strategy(entropy, sampling_coef, min_samples=0, max_samples=10000):
  """
    Computes the number of samples to draw using a linear interpolation with a dampened slope. A positive sampling_coef
    produces maximum samples at high entropy while a negative sampling_coef produces maximum samples at low entropy.
    This is achieved by applying a damping factor to the linear term, based on the affine transformation principle.
    The mapping is defined as:
      factor = 0.5 + sampling_coef * 0.5 * (entropy - 0.5)
      samples = min_samples + (max_samples - min_samples) * factor
    Args:
        entropy: A scalar or array-like value representing normalized entropy in [0, 1].
        sampling_coef: A scalar in [-1, 1] that directs the behavior.
        min_samples: The minimum number of samples.
        max_samples: The maximum number of samples.
    Returns:
        An integer number of samples within [min_samples, max_samples].
    """
  samples_range = max_samples - min_samples
  factor = 0.5 + sampling_coef * 0.5 * (entropy - 0.5)
  samples = min_samples + samples_range * factor
  return int(np.clip(samples, min_samples, max_samples))


import numpy as np


def sampling_strategy(entropy, sampling_coef, min_samples=0, max_samples=128):
  normalized_entropy = abs(entropy)

  if sampling_coef >= 0:
    samples = min_samples + (max_samples - min_samples) * (normalized_entropy) * sampling_coef
  else:
    samples = max_samples - (max_samples - min_samples) * (normalized_entropy) * abs(sampling_coef)

  samples = int(np.clip(samples, min_samples, max_samples))

  return samples


if __name__ == "__main__":
  entropy_values = np.linspace(-10, 10, 1000)  # Entropy range
  entropy_coeff_values = np.linspace(-1, 0, 10)  # Coefficients to test
  test(entropy_values, entropy_coeff_values)

  entropy_values = np.linspace(-10, 10, 1000)  # Entropy range
  entropy_coeff_values = np.linspace(0, 1, 10)  # Coefficients to test

  test(entropy_values, entropy_coeff_values)
