.PHONY: install data train train-quick test lint serve docker-build docker-run

install:        ## Install all dependencies (service + training + dev tools)
	uv sync --extra train
	uv run pre-commit install

data:           ## Download the dataset and verify its checksum
	uv run cs-download

train:          ## Full training run: scorecard + challengers, promotes to champion
	uv run cs-train

train-quick:    ## 10% sample, scorecard only, does not touch the champion
	uv run cs-train --sample-frac 0.1 --skip-challengers --no-promote

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

serve:          ## Run the API locally with reload
	uv run uvicorn credit_scorecard_service.api:app --reload --port 8000

docker-build:
	docker build -t credit-scorecard:local .

docker-run: docker-build
	docker run --rm -p 8000:8000 credit-scorecard:local
