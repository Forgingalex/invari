# Contributing to INVARI

Thank you for your interest in contributing to INVARI! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/invari.git
   cd invari
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   make install-dev
   ```

3. **Set up web frontend:**
   ```bash
   cd web
   npm install
   ```

## Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **JavaScript**: Follow ESLint rules (configured in package.json)
- **Type hints**: Use type annotations for all Python functions
- **Docstrings**: Use Google-style docstrings

## Testing

- Write tests for all new functionality
- Ensure all tests pass: `make test`
- Aim for >80% code coverage

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `make test`
4. Format code: `make format`
5. Lint code: `make lint`
6. Commit with descriptive messages
7. Push and create a pull request

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update API documentation if endpoints change

## Questions?

Open an issue or contact the maintainers.

