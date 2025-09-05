# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the PSA PREGRADER project - a Python-based application that appears to be in early development stages.

## Development Environment

- **Language**: Python
- **Virtual Environment**: `venv/` (already created)
- **Platform**: Windows (win32)

## Common Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies (when requirements.txt is created)
pip install -r requirements.txt
```

### Development Workflow
Since this is a new project, common Python development commands will likely include:

```bash
# Run the application (to be determined based on main entry point)
python main.py  # or appropriate entry point

# Install development dependencies (when available)
pip install -r requirements-dev.txt

# Run tests (when test framework is chosen)
pytest  # or python -m unittest

# Code formatting (when chosen)
black .  # or other formatter

# Linting (when chosen)
flake8 .  # or pylint/ruff
```

## Project Structure

Currently minimal structure:
- `venv/` - Python virtual environment
- `.claude/` - Claude Code configuration

## Notes for Development

- This project is in early development with minimal structure
- Virtual environment is already set up
- No main application files, tests, or configuration files exist yet
- When adding dependencies, create `requirements.txt` for production and `requirements-dev.txt` for development dependencies
- Consider adding a `.gitignore` file for Python projects when initializing git