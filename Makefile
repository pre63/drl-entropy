default: trial


trial:
	@echo "Usage: make trial model=ppo trials=30 envs=4"
	@mkdir -p .logs
	@PYTHONPATH=. python trial.py $(model) | tee -a .logs/sb3-$(date)


board:
	@mkdir -p .logs
	@PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006


install:
	@sudo apt-get -y install swig || brew install swig
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p .logs


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

