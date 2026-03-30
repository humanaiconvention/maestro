.PHONY: test test-cov smoke-test clean

test:
	pytest

test-cov:
	pytest --cov=semantic_grounding --cov-report=term-missing

smoke-test:
	python scripts/smoke_test.py

clean:
	rm -rf tmp/ .pytest_cache .coverage htmlcov/
