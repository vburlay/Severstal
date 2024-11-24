python = python-env/bin/python
pip = python-env/bin/pip
setup:
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt
run:
	$(python) main.py
mlflow:
	python-env/bin/mlflow ui
test:
	$(python) -m pytest


