import numpy as np
import matplotlib.pyplot as plt

def compute_sampling_parameters_gradual_linear(entropy, entropy_coeff, min_samples=0, max_samples=10000):
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


# Updated parameters for the plot
entropy_values = np.linspace(-10, 10, 1000)  # Entropy range
entropy_coeff_values = np.linspace(-1, 1, 10)  # Coefficients to test
min_samples = 0
max_samples = 10000

# Plotting
plt.figure(figsize=(10, 6))
for coeff in entropy_coeff_values:
    if coeff == 0:  # Skip 0 to avoid division by zero
        continue
    samples = [compute_sampling_parameters_gradual_linear(e, coeff, min_samples, max_samples) for e in entropy_values]
    plt.plot(entropy_values, samples, label=f"Entropy Coeff: {coeff:.2f}")

plt.title("Gradual Linear Sampling Distribution (Max/Min at Zero Entropy)")
plt.xlabel("Entropy")
plt.ylabel("Number of Samples")
plt.legend()
plt.grid()
plt.show()

