SHELL := /bin/sh

default: trial

timesteps = 100000 # Default number of timesteps per trial
trials = 10 # Default number of trials per model
envs = 4 # Default number of environments per trial
model = ppo # Default model to train

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

sync:
	aws s3 sync ./.logs s3://entrpo/.logs || true

down:
	aws s3 sync s3://entrpo/.logs ./.logs || true

clean:
	@rm -rf __pycache__/
	@rm -rf .venv

train-zoo:
	@echo "Usage: make train-zoo model=ppo envs=4"
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@. .venv/bin/activate && PYTHONPATH=. python zoo/train.py --model=$(model) --envs=$(envs) | tee -a .logs/zoo-$(model)-$(shell date +"%Y%m%d").log
