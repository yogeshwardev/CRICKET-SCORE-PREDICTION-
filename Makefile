.PHONY: install prepare train test serve dashboard
install:
	python -m pip install -e '.[dev]'
prepare:
	crease prepare
train:
	crease train --trials 6 --iterations 400
test:
	python -m pytest -q
serve:
	crease serve
dashboard:
	cd frontend && npm run dev
