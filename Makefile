.PHONY: install prepare train test experiments report promote serve dashboard export
install:
	python -m pip install -e '.[dev,sequence]'
prepare:
	crease prepare
train:
	crease train --trials 30 --iterations 1500
test:
	python -m pytest -q
experiments:
	python scripts/experiments/feature_ablation.py
	python scripts/experiments/sequence_model.py
	python scripts/experiments/error_analysis.py
report:
	python scripts/render_report.py
export:
	python scripts/export_dashboard.py
serve:
	crease serve
dashboard:
	cd frontend && npm run dev
