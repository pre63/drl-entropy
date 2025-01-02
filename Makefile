default: trial
models = ppo trpo entrpo


trial:
	@echo "Usage: make trial model=ppo trials=30 envs=4"
	@mkdir -p .logs
	@PYTHONPATH=. python trial.py $(model) $(trials) $(envs) | tee -a .logs/sb3-$(shell date +"%Y%m%d").log


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
		done; \
		i=$$((i + 1)); \
	done


install:
	@sudo apt-get -y install swig || brew install swig
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p .logs


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

