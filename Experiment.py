import os
import json
import pandas as pd
from datetime import datetime
import torch


class Experiment:
  @staticmethod
  def save_model(model, folder):
    """Save the model's state dictionary."""
    model_path = os.path.join(folder, "model.pth")
    torch.save(model.state_dict(), model_path)
    return model_path

  @staticmethod
  def save_config(model, folder):
    """Save the model's parameters."""
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as config_file:
      json.dump(model.params, config_file, indent=4)
    return config_path

  @staticmethod
  def save_history(model, folder):
    """Save the full history of metrics."""
    history_path = os.path.join(folder, "history.json")
    with open(history_path, "w") as history_file:
      json.dump(model.get_history(), history_file, indent=4)
    return history_path

  @staticmethod
  def save(model, base_dir="results"):
    """
    Save the model, configuration, metrics, and history.
    Log the experiment results to a CSV file.
    """
    # Create unique directory for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(base_dir, f"{model.name}_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)

    # Save model, configuration, and history
    Experiment.save_model(model, result_folder)
    Experiment.save_config(model, result_folder)
    Experiment.save_history(model, result_folder)

    # Calculate metrics
    metrics = model.get_metrics()
    metrics["Result Folder"] = result_folder

    # Log results to Experiments.csv
    csv_path = "Experiments.csv"
    if not os.path.exists(csv_path):
      results_df = pd.DataFrame(columns=metrics.keys())
    else:
      results_df = pd.read_csv(csv_path)

    # Use pd.concat instead of append
    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
    results_df.to_csv(csv_path, index=False)

    print(f"Experiment logged in folder: {result_folder}\n")
    print(f"    make metrics folder={result_folder}\n")



# Example Usage
if __name__ == "__main__":
  from Specs import ModelSpec

  # Example Model Class inheriting BaseModel
  class ExampleModel(ModelSpec):
    def __init__(self, params):
      super(ExampleModel, self).__init__(params)
      self.fc = torch.nn.Linear(4, 1)

  # Create an example model
  model = ExampleModel(params={"gamma": 0.99, "learning_rate": 0.001})

  # Log individual steps
  model.log_step(reward=10, step_time=0.02)
  model.log_step(reward=12, step_time=0.018)

  # Log an episode
  model.log_episode(rewards=[10, 12, 15], step_times=[0.02, 0.018, 0.021], total_obs=100, success=1)

  # Save the experiment
  Experiment.save(model)
