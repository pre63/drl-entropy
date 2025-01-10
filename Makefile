SHELL := /bin/sh

n_jobs = 2 # Default number of jobs to run in parallel
envs = 4 # Default number of environments to train on
model = ppo # Default model to train
optimize = False # Default to not optimize hyperparameters

zoology = entrpo entrpor trpor trpo ppo tqc sac
zoologyenvs = Pendulum-v1 Ant-v5 Humanoid-v3 InvertedDoublePendulum-v5 LunarLanderContinuous-v3 RocketLander-v0

default: install

board:
	@mkdir -p .logs
	@. .venv/bin/activate && PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006

ubuntu:
	# if not ubuntu exit
	@lsb_release -a | grep "Ubuntu" || exit 1
	@sudo apt-get update
	@sudo apt-get -y install python3-dev swig build-essential cmake
	@sudo apt-get -y install python3.12-venv python3.12-dev
	@sudo apt-get -y install swig python-box2d

mac:
	# if not mac exit
	@sw_vers | grep "Mac" || exit 1
	@brew install python@3.12 box2d swig 

venv:
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt

install: ubuntu mac venv

clean:
	@echo "Cleaning up"
	@rm -rf __pycache__/
	@rm -rf .venv

train:
	@echo "Will train model $(model) on environment $(env) and we will optimize hyperparameters: $(optimize)"
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@. .venv/bin/activate && PYTHONPATH=. python zoo/train.py --model=$(model) --env=$(env) --optimize=$(optimize) --envs=$(envs) --n_jobs=$(n_jobs) | tee -a .logs/zoo-$(model)-$(env)-$(shell date +"%Y%m%d").log

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
	@while true; do \
		while read -r line; do \
			model=$$(echo $$line | cut -d':' -f1); \
			envs=$$(echo $$line | cut -d':' -f2); \
			for env in $$envs; do \
				$(MAKE) train model=$$model env=$$env optimize=True || echo "Training failed for $$model on $$env"; \
			done; \
		done < configs.txt; \
	done
