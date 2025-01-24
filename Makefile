SHELL := /bin/bash

OS := $(shell uname -s)

n_jobs=10 # Default number of jobs to run in parallel
envs=10 # Default number of environments to train on
model=trpo # Default model to train
optimize=False # Default to not optimize hyperparameters
trials=1000 # Default number of trials for hyperparameter optimization
n_timesteps=0 # Default number of timesteps to train for
env=Humanoid-v5 # Default environment to train on
configs=configs.txt # Default configuration file

zoology=entrpo entrpor trpo entrpohigh entrpolow
zoologyenvs=Ant-v5 Humanoid-v5 InvertedDoublePendulum-v5

default: install

board:
	@mkdir -p .logs
	@. .venv/bin/activate && PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006

ubuntu:
	@if [ "$(OS)" != "Linux" ]; then \
		echo "Not a Linux system, skipping Ubuntu setup."; \
	elif ! command -v lsb_release > /dev/null; then \
		echo "lsb_release not found, skipping Ubuntu setup."; \
	elif ! lsb_release -a 2>/dev/null | grep -q "Ubuntu"; then \
		echo "Not an Ubuntu system, skipping."; \
	else \
		echo "Running Ubuntu setup..."; \
		sudo apt-get update && \
		sudo apt-get -y install python3-dev swig build-essential cmake && \
		sudo apt-get -y install python3.12-venv python3.12-dev && \
		sudo apt-get -y install swig python-box2d; \
	fi

mac:
	@if [ "$(OS)" != "Darwin" ]; then \
		echo "Not a macOS system, skipping macOS setup."; \
	elif ! command -v sw_vers > /dev/null; then \
		echo "sw_vers not found, skipping macOS setup."; \
	elif ! sw_vers | grep -q "macOS"; then \
		echo "Not a macOS system, skipping."; \
	else \
		echo "Running macOS setup..."; \
		brew install python@3.12 box2d swig; \
	fi

venv:
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

install: ubuntu mac venv

fix:
	@echo "Will run black and isort on modified, added, untracked, or staged Python files"
	@changed_files=$$(git diff --name-only --diff-filter=AM | grep '\.py$$'); \
	untracked_files=$$(git ls-files --others --exclude-standard | grep '\.py$$'); \
	staged_files=$$(git diff --name-only --cached | grep '\.py$$'); \
	all_files=$$(echo "$$changed_files $$untracked_files $$staged_files" | tr ' ' '\n' | sort -u); \
	if [ ! -z "$$all_files" ]; then \
		. .venv/bin/activate && isort --multi-line=0 --line-length=100 $$all_files && black .; \
	else \
		echo "No modified, added, untracked, or staged Python files"; \
	fi

clean:
	@echo "Cleaning up"
	@rm -rf __pycache__/
	@rm -rf .venv

train:
	@echo "Will train model $(model) on environment $(env) and we will optimize hyperparameters: $(optimize)"
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@mkdir -p ".optuna-zoo/$(model)_$(env)"
	@. .venv/bin/activate; PYTHONPATH=. python -u zoo/train.py --model=$(model) --env=$(env) --optimize=$(optimize) --n_jobs=$(n_jobs) --trials=$(trials) 2>&1 | tee -a .logs/zoo-$(model)-$(env)-$(shell date +"%Y%m%d").log

train-zoo:
	@echo "Will train all models in zoo"
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@mkdir -p .logs/tensorboard
	@for env in $(zoologyenvs); do \
		for model in $(zoology); do \
			$(MAKE) train model=$$model env=$$env optimize=True || true; \
		done; \
	done
nightly:
	@$(MAKE) fix
	@while true; do \
		while read -r line; do \
			zmodel=$$(echo $$line | cut -d':' -f1); \
			zenvs=$$(echo $$line | cut -d':' -f2); \
			for env in $$zenvs; do \
				echo "Launching training for $$zmodel on $$env with $$envs parallel jobs..."; \
				for i in $$(seq 1 $(envs)); do \
					$(MAKE) train model=$$zmodel env=$$env optimize=True & \
				done; \
				wait; \
			done; \
		done < $(configs); \
	done

train-eval:
	@echo "Will evaluate model $(model) on environment $(env)"
	@for env in $(zoologyenvs); do \
			. .venv/bin/activate; PYTHONPATH=. python -u zoo/train-eval.py --model=$(model) --env=$(env) --n_timesteps=$(n_timesteps) 2>&1 | tee -a .logs/eval-$(model)-$(env)-$(shell date +"%Y%m%d").log; \
	done

train-eval-all:
	@echo "Will evaluate all models in zoo"
	@for model in $(zoology); do \
		for env in $(zoologyenvs); do \
			$(MAKE) train-eval model=$$model env=$$env || true; \
		done; \
	done

train-eval-plot:
	@$(MAKE) fix
	@echo "Will evaluate model $(model) on environment $(env) and generate a plot"
	@. .venv/bin/activate; PYTHONPATH=. python -u zoo/train_eval_plot.py

list:
	@# List all zoo models and environments combinations
	@echo "Listing all model and environment combinations:"
	@for env in $(zoologyenvs); do \
		for model in $(zoology); do \
			echo "$$env - $$model"; \
		done; \
	done
	@# Calculate and print the total number of combinations
	@zoologyenvs_count=$$(echo "$(zoologyenvs)" | wc -w); \
	zoology_count=$$(echo "$(zoology)" | wc -w); \
	trials=$(trials); \
	total_combinations=$$(echo "$$zoologyenvs_count * $$zoology_count * $$trials" | bc); \
	echo "Total number of combinations: $$total_combinations"
