
experiment:
	@python Experiment.py


trpo:
	@PYTHONPATH=. python Models/TRPO.py $(FROM) | tee -a .logs/trpo.logs


trpo3:
	@PYTHONPATH=. python Models/TRPO3.py | tee -a .logs/trpo3.logs


entropy:
	@PYTHONPATH=. python Models/EnTRPO.py | tee -a .logs/entropy.logs


trace:
	@PYTHONPATH=. python Models/EnTRPOTrace.py | tee -a .logs/trace.logs


run:
	@PYTHONPATH=. python run.py | tee -a .logs/run.logs


metrics:
	@PYTHONPATH=. python Metrics.py $(folder)

sb3:
	@PYTHONPATH=. python sb3.py $(model) | tee -a .logs/sb3.logs

board:
	@PYTHONPATH=. tensorboard --logdir=./.logs/tensorboard/ --port 6006

install:
	@sudo apt-get -y install swig || brew install swig
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p .logs


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

