SHELL := /bin/sh

default: trial

models = ppo trpo entrpo entrpo # Default models to train
launches = 4 # Default number of launches per model
timesteps = 10000 # Default number of timesteps per trial
trials = 10 # Default number of trials per model
envs = 4 # Default number of environments per trial
model = ppo # Default model to train

board:
	@mkdir -p .logs
	@. .venv/bin/activate && PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006

nightly:
	@echo "Usage: make nightly models='entrpor entrpo trpo ppo' launches=10"
	@mkdir -p .logs
	@echo "Starting nightly training for models: $(models)"
	@i=1; while true; do \
		for model in $(models); do \
			echo "Launching $(launches) concurrent trials for model=$$model"; \
			for j in $$(seq 1 $(launches)); do \
				{ $(MAKE) trial model=$$model trials=1 envs=10 timesteps=$(timesteps) | tee -a .logs/sb3-$$model-$$j-$(shell date +"%Y%m%d").log & }; \
				$(MAKE) sync; \
			done; \
		done; \
		wait; \
		i=$$((i + 1)); \
	done

trial:
	@echo "Usage: make trial model=ppo trials=10 envs=4"
	@mkdir -p .logs
	@mkdir -p .optuna
	@rm -f .optuna/storage.lock || true
	@echo "Starting trial for model=$(model) with $(trials) trials and $(envs) environments..."
	@. .venv/bin/activate && PYTHONPATH=. python trial.py --model $(model) --trials $(trials) --envs $(envs) --timesteps $(timesteps) | tee -a .logs/sb3-$(model)-$(shell date +"%Y%m%d").log
	$(MAKE) sync

sbx:
	@echo "Usage: make sbx model=ppo trials=10 timesteps=1000000 envs=4"
	@mkdir -p .logs
	@mkdir -p .optuna-sbx
	@rm -f .optuna-sbx/storage.lock || true
	@echo "Starting trial for model=$(model) with $(trials) trials and $(envs) environments..."
	@. .venv/bin/activate && PYTHONPATH=. python trial-sbx.py --model $(model) --trials $(trials) --timesteps $(timesteps) --envs $(envs) | tee -a .logs/sbx-$(model)-$(shell date +"%Y%m%d").log

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
	aws s3 sync ./.models s3://entrpo/.models || true
	aws s3 cp ./Experiments.csv s3://entrpo/ || true
	aws s3 cp ./Trials.csv s3://entrpo/ || true

down:
	aws s3 sync s3://entrpo/.logs ./.logs || true
	aws s3 sync s3://entrpo/.models ./.models || true
	aws s3 cp s3://entrpo/Experiments.csv ./ || true
	aws s3 cp s3://entrpo/Trials.csv ./ || true

clean:
	@rm -rf __pycache__/
	@rm -rf .venv
