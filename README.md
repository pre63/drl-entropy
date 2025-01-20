# Deep Reinforcement Learning Zoo 🐾

Welcome to the **Deep RL Zoo**, where cutting-edge reinforcement learning (RL) meets experimentation! This repo helps you train, evaluate, and optimize various RL models across multiple environments with minimal hassle. Dive in and start exploring the frontier of AI-powered agents! 🚀

---

## 📦 Features
- **One-stop RL training:** Train state-of-the-art models like `TRPO`, `ENTRPO`, and more across diverse environments.
- **Hyperparameter Optimization:** Perform large-scale searches to find the best settings for your agents.
- **Seamless Automation:** Automate nightly experiments and evaluate performance effortlessly.
- **Flexible Setup:** Ready for both Ubuntu and macOS users with straightforward environment creation.
- **Integrated Visualization:** Use TensorBoard for clear insights into training dynamics.

---

## 🛠️ Quick Start

1. **Install everything:**
   ```bash
   make install
   ```
   Installs dependencies, sets up a virtual environment, and prepares your system for training.

2. **Train a model:**
   ```bash
   make train model=trpo env=Humanoid-v5
   ```
   Trains the `TRPO` model on the `Humanoid-v5` environment.

3. **Run hyperparameter search:**
   ```bash
   make nightly envs=10 n_jobs=5 trials=100 & disown
   ```
   Kicks off a large-scale hyperparameter optimization across all models and environments.

4. **Visualize training progress:**
   ```bash
   make board
   ```
   Launches TensorBoard on `http://localhost:6006`.

5. **List all experiments:**
   ```bash
   make list
   ```
   Displays all possible model-environment combinations.

---

## 🧪 Advanced Usage

### Evaluate Models
Run evaluations on trained models:
```bash
make train-eval model=trpo env=Ant-v5
```

### Evaluate and Plot
Generate performance plots for trained models:
```bash
make train-eval-plot model=trpo env=Humanoid-v5
```

### Train the Entire Zoo
Train all models across all environments:
```bash
make train-zoo
```

### Clean Everything
Start fresh by cleaning up the environment:
```bash
make clean
```

---

## 🌍 Supported Models and Environments

### Models
- **TRPO**
- **ENTRPO**
- **ENTRPO High/Low**

### Environments
- **Ant-v5**
- **Humanoid-v5**
- **InvertedDoublePendulum-v5**

---

## 🤖 Why Use This Repo?

- **Scalability:** Easily scale experiments across multiple environments and hyperparameter settings.
- **Automation:** Save time with robust automation tools for nightly runs.
- **Insights:** Use TensorBoard for detailed performance analysis.

---

🎉 Happy Reinforcement Learning! 🚀