# Chaos to Knowledge: Entropy in Continuous Control

This repository supports the research paper *Chaos to Knowledge: Entropy in Continuous Control*. It serves as a comprehensive platform for training, evaluating, and optimizing reinforcement learning (RL) algorithms in continuous control environments, particularly focusing on entropy-based modifications to Trust Region Policy Optimization (TRPO). The repository provides tools to automate experiments, optimize hyperparameters, and visualize performance, all while enabling seamless exploration of advanced RL strategies.

[Read the generated report here.](report.pdf)

## Features

This repository is designed to streamline reinforcement learning experimentation. It supports the training of state-of-the-art RL models, including TRPO and entropy-regularized TRPO variants, across diverse continuous control environments. Hyperparameter tuning is a core feature, enabling users to conduct large-scale searches for optimal configurations. Automation capabilities allow for effortless scheduling of nightly experiments, ensuring robust and efficient exploration of various setups. TensorBoard integration further provides a detailed visualization of training progress and performance dynamics, enhancing the interpretability of experimental outcomes.

## Quick Start

The repository includes a Makefile for streamlined setup and execution. To prepare the environment, install dependencies, and set up a virtual environment, run `make install`. Training a model is as simple as specifying the algorithm and environment, for example, `make train model=trpo env=Humanoid-v5`. Users can also initiate large-scale hyperparameter optimization with commands like `make nightly envs=10 n_jobs=5 trials=100`, allowing extensive exploration across multiple configurations and environments. To visualize training progress, launch TensorBoard with `make board`, which provides an intuitive interface for monitoring metrics and trends.

## Advanced Usage

The platform accommodates advanced experimentation needs. Evaluation of trained models can be performed using commands like `make train-eval model=trpo env=Ant-v5`, while generating performance plots is handled with `make train-eval-plot model=trpo env=Humanoid-v5`. For comprehensive testing, the command `make train-zoo` trains all supported models across all available environments, enabling a holistic assessment of algorithmic performance. Cleaning up the environment to ensure a fresh start can be achieved with `make clean`.

## Supported Models and Environments

The repository supports several reinforcement learning models, including standard TRPO, entropy-regularized TRPO, and high/low entropy variants. The environments span tasks with varying complexities, such as Ant-v5, Humanoid-v5, and Inverted Double Pendulum. Each environment presents unique challenges in observation and action space dimensions, making them ideal benchmarks for evaluating entropy-based exploration strategies.

## Research Context

The research paper examines the impact of entropy-based exploration on the stability and convergence of learning in continuous control settings. By embedding entropy into trust-region constraints and selectively reusing samples via replay mechanisms, the study explores whether these modifications enhance sample efficiency and mitigate premature convergence. The experiments span environments of varying complexity, from intermediate pendulum systems to the intricate dynamics of humanoid locomotion, providing a diverse testing ground for the proposed approaches.

The methodology focuses on how entropy affects policy optimization in large state-action spaces. Techniques like adding entropy bonuses to the surrogate objective aim to balance exploration with stability, while integrating entropy into trust-region constraints offers adaptive flexibility in policy updates. Replay mechanisms are also explored, with strategies triggering sample reuse based on entropy levels, either to guard against early convergence or to consolidate knowledge during high-entropy exploration phases.

## Methodology and Automation

This repository employs automated hyperparameter tuning frameworks to ensure fair and consistent comparisons across models and environments. Each algorithm-environment pair undergoes independent tuning, with performance summarized using mean and variance of episodic returns. Automation tools allow for systematic experimentation, minimizing manual intervention and ensuring reproducibility.

## Why This Repository?

This platform is a one-stop solution for exploring entropy-driven reinforcement learning strategies in continuous control. It provides the tools and infrastructure needed to validate and expand upon the findings presented in *Chaos to Knowledge: Entropy in Continuous Control*, offering a solid foundation for advancing research in entropy-regularized RL methods.


