export PYTHONHASHSEED = 1

run:
	venv/bin/python src/main.py

setup:
	python3 -m virtualenv --always-copy --system-site-packages venv
	venv/bin/pip install -r requirements.txt

format:
	venv/bin/python -m black src

clean:
	rm -rf venv

.PHONY: run
