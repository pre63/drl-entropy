
experiment:
	@python Experiment.py

trpo:
	@PYTHONPATH=. python Models/TRPO.py

entropy:
	@PYTHONPATH=. python Models/EnTRPO.py

trace:
	@PYTHONPATH=. python Models/EnTRPOTrace.py

all_experiments:
	@PYTHONPATH=. python Models/TRPO.py
	@PYTHONPATH=. python Models/EnTRPO.py
	@PYTHONPATH=. python Models/EnTRPOTrace.py

install:
	@test -d .venv || python3.12 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt

clean:
	@rm -rf __pycache__/
	@rm -rf .venv

