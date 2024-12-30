import os
import json
import pandas as pd
from datetime import datetime
import torch


class Experiment:
  """
  Handles model saving, configuration saving, history saving, 
  and logging the main training metrics to a CSV file.
  """

  @staticmethod
  def save_model(model, folder):
    model_path = os.path.join(folder, "model.pth")
    torch.save(model.state_dict(), model_path)
    return model_path

  @staticmethod
  def save_config(model, folder):
    config_data = {
        "model_name": model.name,
        **model.params
    }
    config_path = os.path.join(folder, "config.json")
    with open(config_path, "w") as cfg:
      json.dump(config_data, cfg, indent=4)
    return config_path

  @staticmethod
  def save_history(model, folder):
    """
    Save all logs: step_logs, episode_logs, and eval_logs.
    """
    data = {
        "step_logs": model.get_step_metrics(),
        "episode_logs": model.get_episode_metrics(),
        "eval_logs": model.get_eval_metrics(),
        "loss_log": model.get_loss_metrics()
    }
    hist_path = os.path.join(folder, "history.json")
    with open(hist_path, "w") as h:
      json.dump(data, h, indent=4)
    return hist_path

  @staticmethod
  def save(model, env, base_dir="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(base_dir, f"{model.name}_{timestamp}")
    os.makedirs(result_folder, exist_ok=True)

    Experiment.save_model(model, result_folder)
    Experiment.save_config(model, result_folder)
    Experiment.save_history(model, result_folder)

    metrics = model.get_training_summary()
    metrics["Result Folder"] = result_folder
    if env is not None:
      metrics["Environment"] = getattr(env, "name", "UnknownEnv")
    else:
      metrics["Environment"] = "UnknownEnv"

    csv_path = "Experiments.csv"
    if not os.path.exists(csv_path):
      results_df = pd.DataFrame(columns=metrics.keys())
    else:
      results_df = pd.read_csv(csv_path)

    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
    results_df.to_csv(csv_path, index=False)

    print(f"Experiment logged in folder: {result_folder}\n")
    print(f"    make metrics folder={result_folder}\n")

    return result_folder


# Example Usage
if __name__ == "__main__":
  from Specs import ModelSpec

  # Example Model Class inheriting BaseModel
  class ExampleModel(ModelSpec):
    def __init__(self, **params):
      super(ExampleModel, self).__init__(**params)
      self.fc = torch.nn.Linear(4, 1)

  # Create an example model
  model = ExampleModel(params={"gamma": 0.99, "learning_rate": 0.001})

  # Log individual steps
  model.log_step(reward=10, step_time=0.02)
  model.log_step(reward=12, step_time=0.018)

  # Log an episode
  model.log_episode(episode_rewards=100, success=1, episode_steps=10, avg_step_time=0.02)

  # Save the experiment
  Experiment.save(model, env=None)
