default: trial

models = ppo trpo entrpo entrpor	# Default models to train
launches = 4  # Default number of launches per model

board:
	@mkdir -p .logs
	@PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006

nightly:
	@echo "Usage: make nightly models='entrpor entrpo trpo ppo' launches=10"
	@mkdir -p .logs
	@echo "Starting nightly training for models: $(models)"
	@i=1; while true; do \
		for model in $(models); do \
			echo "Launching $(launches) concurrent trials for model=$$model"; \
			for j in $$(seq 1 $(launches)); do \
				{ $(MAKE) trial model=$$model trials=1 envs=10 | tee -a .logs/sb3-$$model-$$j-$(shell date +"%Y%m%d").log & }; \
			done; \
		done; \
		wait; \
		$(MAKE) sync; \
		i=$$((i + 1)); \
	done

trial:
	@echo "Usage: make trial model=ppo trials=30 envs=4"
	@mkdir -p .logs
	@mkdir -p .optuna
	@rm -f .optuna/storage.lock || true
	@echo "Starting trial for model=$(model) with $(trials) trials and $(envs) environments..."
	@PYTHONPATH=. python trial.py $(model) $(trials) $(envs) | tee -a .logs/sb3-$(model)-$(shell date +"%Y%m%d").log
	$(MAKE) sync

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
	aws s3 sync ./.logs s3://entrpo/.logs
	aws s3 sync ./.models s3://entrpo/.models
	aws s3 cp ./Experiments.csv s3://entrpo/ || true
	aws s3 cp ./Trials.csv s3://entrpo/ || true

clean:
	@rm -rf __pycache__/
	@rm -rf .venv
