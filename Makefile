init:
	pip install -r requirements.txt

test:
	python -m unittest discover -s tests -v