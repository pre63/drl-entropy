import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, result_path):
        self.result_path = result_path
        self.metrics_file = os.path.join(result_path, "Experiments.csv")
        self.config_file = os.path.join(result_path, "config.json")

        if not os.path.exists(self.metrics_file):
            raise FileNotFoundError(f"Metrics file not found at {self.metrics_file}")
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found at {self.config_file}")

        # Load metrics and configuration
        self.metrics = pd.read_csv(self.metrics_file)
        with open(self.config_file, "r") as f:
            self.config = json.load(f)

    def plot_metric(self, metric_name, title=None, xlabel="Episodes", ylabel=None):
        """
        Plot a specific metric over episodes.
        """
        if metric_name not in self.metrics.columns:
            raise ValueError(f"Metric '{metric_name}' not found in the metrics file.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics.index, self.metrics[metric_name], label=metric_name)
        plt.title(title or f"{metric_name} over Episodes")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric_name)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_all(self):
        """
        Plot all metrics available in the metrics file.
        """
        for metric in self.metrics.columns:
            if metric != "Episode":  # Skip episode index column
                self.plot_metric(metric_name=metric, title=f"{metric} over Episodes")

    def summary(self):
        """
        Print a summary of the metrics and configuration.
        """
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        print("\nMetrics Overview:")
        print(self.metrics.describe())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Metrics.py [result_path]")
        sys.exit(1)

    result_path = sys.argv[1]
    try:
        metrics = Metrics(result_path)
        metrics.summary()
        metrics.plot_all()
    except Exception as e:
        print(f"Error: {e}")
