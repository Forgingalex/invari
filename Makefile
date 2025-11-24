.PHONY: help install install-dev test run-api run-web run-docker clean format lint

help:
	@echo "INVARI - Sensor Fusion Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make run-api       - Start FastAPI server"
	@echo "  make run-web       - Start React development server"
	@echo "  make run-docker    - Start services with Docker Compose"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with flake8"
	@echo "  make clean         - Clean generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-web:
	cd web && npm start

run-docker:
	docker-compose up --build

format:
	black src/ api/ scripts/ tests/

lint:
	flake8 src/ api/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	rm -rf web/node_modules web/build

