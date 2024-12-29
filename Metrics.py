import os
import sys
import json
import math
import matplotlib.pyplot as plt


class Metrics:
  def __init__(self, result_path):
    self.result_path = result_path
    self.config_file = os.path.join(result_path, "config.json")
    self.history_file = os.path.join(result_path, "history.json")

    if not os.path.exists(self.config_file):
      raise FileNotFoundError(f"Configuration file not found at {self.config_file}")
    if not os.path.exists(self.history_file):
      raise FileNotFoundError(f"History file not found at {self.history_file}")

    # Load configuration and history data
    with open(self.config_file, "r") as f:
      self.config = json.load(f)
    with open(self.history_file, "r") as f:
      self.history = json.load(f)

  def plot_all_metrics(self):
    """
    Plot all metrics available in the history file in a grid layout.
    """
    metrics = list(self.history.keys())
    num_metrics = len(metrics)
    cols = 2  # Number of columns in the grid
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), constrained_layout=True)

    # Flatten axes for easy indexing
    axes = axes.flatten() if rows > 1 else [axes]

    for idx, metric in enumerate(metrics):
      ax = axes[idx]
      ax.plot(range(len(self.history[metric])), self.history[metric], label=metric, color="blue")
      ax.set_title(f"{metric} Over Time", fontsize=12)
      ax.set_ylabel(metric, fontsize=10)
      ax.set_xlabel("Timesteps", fontsize=10)
      ax.legend()
      ax.grid()

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
      axes[idx].axis("off")

    plt.suptitle("Metrics Overview", fontsize=16)
    plt.show()

  def summary(self):
    """
    Print a summary of the configuration and available metrics.
    """
    print("\nConfiguration:")
    for key, value in self.config.items():
      print(f"  {key}: {value}")

    print("\nMetrics Overview:")
    print("Available metrics in history:")
    for metric in self.history.keys():
      print(f"  {metric}")


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python Metrics.py [result_path]")
    sys.exit(1)

  result_path = sys.argv[1]
  try:
    metrics = Metrics(result_path)
    metrics.summary()
    metrics.plot_all_metrics()
  except Exception as e:
    print(f"Error: {e}")
