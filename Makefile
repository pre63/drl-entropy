default: trial
models = ppo trpo entrpo


trial:
	@echo "Usage: make trial model=ppo trials=30 envs=4"
	@mkdir -p .logs
	@PYTHONPATH=. python trial.py $(model) $(trials) $(envs) | tee -a .logs/sb3-$(shell date +"%Y%m%d").log
	$(MAKE) sync


board:
	@mkdir -p .logs
	@PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006


nightly:
	@echo "Usage: make nightly models='entrpor entrpo trpo ppo'"
	@mkdir -p .logs
	@i=1; while true; do \
		for model in $(models); do \
			echo "Running trial for model=$$model"; \
			{ $(MAKE) trial model=$$model trials=10 envs=15 | tee -a .logs/sb3-$$model-$$i-$(shell date +"%Y%m%d").log; } || echo "Error encountered, continuing with next model..."; \
			$(MAKE) sync \
		done; \
		i=$$((i + 1)); \
	done


ubuntu:
	# if not ubuntu exit
	@lsb_release -a | grep "Ubuntu" || exit 1
	@sudo apt-get  update
	@sudo apt-get  -y install python3-dev swig build-essential cmake
	@sudo apt-get  -y install python3.12-venv python3.12-dev
	@sudo apt-get -y install  swig python-box2d


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
	aws s3 cp ./Experiments.csv s3://entrpo/
	aws s3 cp ./Trial.csv s3://entrpo/


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

