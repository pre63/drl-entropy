
experiment:
	@python Experiment.py


trpo:
	@PYTHONPATH=. python Models/TRPO.py $(FROM) | tee -a log/trpo.log


trpo3:
	@PYTHONPATH=. python Models/TRPO3.py | tee -a log/trpo3.log


entropy:
	@PYTHONPATH=. python Models/EnTRPO.py | tee -a log/entropy.log


trace:
	@PYTHONPATH=. python Models/EnTRPOTrace.py | tee -a log/trace.log


run:
	@PYTHONPATH=. python run.py | tee -a log/run.log


metrics:
	@PYTHONPATH=. python Metrics.py $(folder)


install:
	@sudo apt-get -y install swig || brew install swig
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p log


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

