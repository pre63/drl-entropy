
experiment:
	@python Experiment.py


trpo:
	@PYTHONPATH=. python Models/TRPO.py | tee -a log/trpo.log


trpo3:
	@PYTHONPATH=. python Models/TRPO3.py | tee -a log/trpo3.log


entropy:
	@PYTHONPATH=. python Models/EnTRPO.py | tee -a log/entropy.log


trace:
	@PYTHONPATH=. python Models/EnTRPOTrace.py | tee -a log/trace.log


all_experiments:
	@PYTHONPATH=. python Models/TRPO.py | tee -a log/trpo.log
	@PYTHONPATH=. python Models/EnTRPO.py | tee -a log/entropy.log
	@PYTHONPATH=. python Models/EnTRPOTrace.py | tee -a log/trace.log


metrics:
	@PYTHONPATH=. python Metrics.py $(folder)


install:
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@mkdir -p log


clean:
	@rm -rf __pycache__/
	@rm -rf .venv

