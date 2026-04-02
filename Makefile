.PHONY: test clean ingest run demo

test:
	poetry run pytest -q

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf ./src/data/index
	rm -f ./src/data/processed/chunks.jsonl

ingest:
	poetry run classical-quotes ingest --input ./src/data/raw

run:
	PYTHONPATH=src poetry run uvicorn app.main:app --reload

demo:
	poetry run classical-quotes demo
