import numpy as np


def sampling_strategy(entropy, sampling_coef, min_samples=0, max_samples=128):
  # Normalize entropy to [0,1] using a sigmoid transformation
  normalized_entropy = 1 / (1 + np.exp(-entropy))  # Sigmoid function ensures bounded output

  if sampling_coef >= 0:
    # When coef is positive, sample at max when entropy is greater than coef
    samples = max_samples if normalized_entropy >= sampling_coef else min_samples
  else:
    # When coef is negative, invert logic: sample at max when entropy is LESS than |coef|
    samples = max_samples if normalized_entropy <= abs(sampling_coef) else min_samples

  # print summary of the strategy variables
  print(f"Entropy: {entropy:.4f}, Normalized Entropy: {normalized_entropy:.4f}, coefficient: {sampling_coef}, Samples: {samples}")

  return samples
