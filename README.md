# Chaos to Knowledge: Entropy in Continuous Control

This repo supports the paper *Chaos to Knowledge: Entropy in Continuous Control*, where we explore how entropy-based changes to Trust Region Policy Optimization (TRPO) impact learning in continuous control tasks. We test whether these modifications improve sample efficiency, prevent early convergence, and enhance stability across different environments, from simple pendulums to complex humanoid locomotion.

[Read the report here.](Report.pdf)

All commands to run and reproduce the experiments are in the Makefile. Training, evaluation, and hyperparameter tuning are fully automated. Raw evaluation data is stored in `.eval`, hyperparameter configurations in `.hyperparameters`, and fine-tuning results in `.results`, making it easy to inspect and verify outcomes.

## Citation

If you use this work, please cite it as:

```bibtex
@article{green2025chaos,
  title = {Chaos to Knowledge: Entropy in Continuous Control},
  author = {Simon Green and Abdulrahman Altahhan},
  institution = {School of Computing, University of Leeds, UK},
  year = {2025},
  note = {Preprint}
}
```