import os
import sys
import json
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # Replace "TkAgg" with another backend if needed


class Metrics:
    def __init__(self, result_path):
        self.result_path = result_path
        self.config_file = os.path.join(result_path, "config.json")
        self.history_file = os.path.join(result_path, "history.json")

        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found at {self.config_file}")
        if not os.path.exists(self.history_file):
            raise FileNotFoundError(f"History file not found at {self.history_file}")

        with open(self.config_file, "r") as f:
            self.config = json.load(f)
        with open(self.history_file, "r") as f:
            self.history = json.load(f)

        # Expected keys for the new logging structure:
        self.step_logs = self.history.get("step_logs", [])
        self.episode_logs = self.history.get("episode_logs", [])
        self.eval_logs = self.history.get("eval_logs", ([], []))
        self.loss_log = self.history.get("loss_log", [])

    def summary(self):
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        print("\nHistory Contents:")
        if self.step_logs:
            print(f"  Step Logs: {len(self.step_logs)} entries")
        if self.episode_logs:
            print(f"  Episode Logs: {len(self.episode_logs)} entries")
        if self.eval_logs:
            print(f"  Eval Logs: {len(self.eval_logs[0])} entries")
        if self.loss_log:
            print(f"  Loss Logs: {len(self.loss_log)} entries")

    def plot_all_metrics(self):
        """
        Plot all metrics in a grid layout with 3 columns.
        """
        params_str = ", ".join([f"{k}={v}" for k, v in self.config.items()])

        # Determine the total number of plots
        plots = []
        if self.step_logs:
            plots.append(("Step-Based Rewards", self.plot_step_rewards))
        if self.loss_log:
            plots.append(("Actor Loss", self.plot_actor_loss))
            plots.append(("Critic Loss", self.plot_critic_loss))
            plots.append(("KL Divergence", self.plot_kl_div))
        if self.episode_logs:
            plots.append(("Episode Rewards", self.plot_episode_rewards))
            plots.append(("Avg Step Time", self.plot_avg_step_time))
            plots.append(("Episode Success", self.plot_episode_success))
        if self.eval_logs:
            plots.append(("Evaluation Rewards", self.plot_eval_rewards))
            plots.append(("Evaluation Success", self.plot_eval_success))

        num_plots = len(plots)
        cols = 3
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
        axes = axes.flatten()

        for idx, (title, plot_func) in enumerate(plots):
            ax = axes[idx]
            plot_func(ax, title)

        # Hide unused subplots
        for idx in range(len(plots), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(params_str, fontsize=14)
        plt.show()

    def plot_step_rewards(self, ax, title):
        timesteps = [entry["timestep"] for entry in self.step_logs]
        rewards = [entry["reward"] for entry in self.step_logs]
        ax.plot(timesteps, rewards, color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Timesteps", fontsize=10)
        ax.set_ylabel("Rewards", fontsize=10)
        ax.grid(True)

    def plot_actor_loss(self, ax, title):
        actor_losses = [entry["actor_loss"] for entry in self.loss_log]
        batch_indices = range(1, len(actor_losses) + 1)
        ax.plot(batch_indices, actor_losses, color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("Actor Loss", fontsize=10)
        ax.grid(True)

    def plot_critic_loss(self, ax, title):
        critic_losses = [entry["critic_loss"] for entry in self.loss_log]
        batch_indices = range(1, len(critic_losses) + 1)
        ax.plot(batch_indices, critic_losses, color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("Critic Loss", fontsize=10)
        ax.grid(True)

    def plot_kl_div(self, ax, title):
        kl_divs = [entry["kl_divergence"] for entry in self.loss_log]
        batch_indices = range(1, len(kl_divs) + 1)
        ax.plot(batch_indices, kl_divs, color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Batch", fontsize=10)
        ax.set_ylabel("KL Divergence", fontsize=10)
        ax.grid(True)

    def plot_episode_rewards(self, ax, title):
        indices = [log["episode_index"] for log in self.episode_logs]
        returns = [log["episode_rewards"] for log in self.episode_logs]
        ax.plot(indices, returns, color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Rewards", fontsize=10)
        ax.grid(True)

    def plot_avg_step_time(self, ax, title):
        indices = [log["episode_index"] for log in self.episode_logs]
        avg_step_times = [log["avg_step_time"] for log in self.episode_logs]
        ax.plot(indices, avg_step_times, color="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Avg Step Time (s)", fontsize=10)
        ax.grid(True)

    def plot_episode_success(self, ax, title):
        indices = [log["episode_index"] for log in self.episode_logs]
        successes = [log["success"] for log in self.episode_logs]
        ax.plot(indices, successes, color="black", linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Success Flag", fontsize=10)
        ax.grid(True)

    def plot_eval_rewards(self, ax, title):
        rewards, _ = self.eval_logs
        eval_indices = range(1, len(rewards) + 1)
        ax.plot(eval_indices, rewards, color="black", marker="o")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Rewards", fontsize=10)
        ax.grid(True)

    def plot_eval_success(self, ax, title):
        _, successes = self.eval_logs
        eval_indices = range(1, len(successes) + 1)
        ax.plot(eval_indices, successes, color="black", marker="o")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episodes", fontsize=10)
        ax.set_ylabel("Success", fontsize=10)
        ax.grid(True)


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
