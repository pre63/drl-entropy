SHELL := /bin/sh

envs = 4 # Default number of environments per trial
model = ppo # Default model to train
optimize = False # Default to not optimize hyperparameters

zoology = entrpo entrpor trpor trpo ppo tqc sac
zoologyenvs = LunarLanderContinuous-v3 Ant-v3 Humanoid-v3 InvertedDoublePendulum-v3 RocketLander-v0

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

sync:
	aws s3 sync ./.logs s3://entrpo/.logs || true

down:
	aws s3 sync s3://entrpo/.logs ./.logs || true

clean:
	@echo "Cleaning up"
	@rm -rf __pycache__/
	@rm -rf .venv

train:
	@echo "Will train model $(model) on environment $(env) for $(envs) environments and we will optimize hyperparameters: $(optimize)"
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@. .venv/bin/activate && PYTHONPATH=. python zoo/train.py --model=$(model) --envs=$(envs) --env=$(env) --optimize=$(optimize) | tee -a .logs/zoo-$(model)-$(shell date +"%Y%m%d").log

train-zoo:
	@echo "Will train all models in zoo"
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@mkdir -p .logs/tensorboard
	@for model in $(zoology); do \
		for env in $(zoologyenvs); do \
			$(MAKE) train model=$$model env=$$env || true; \
		done; \
	done

train-exp:
	@mkdir -p .logs
	@mkdir -p .optuna-zoo
	@mkdir -p .logs/tensorboard
	
	@$(MAKE) train-zoo models="trpoq trpoq2" 