# GitHub Actions CI/CD Setup Guide

## 🚀 Overview

This repository uses GitHub Actions for continuous integration, testing, and deployment. The workflows automatically run tests, check code quality, build Docker images, and create releases.

## 📋 Workflows

### 1. **Python Tests** (`python-tests.yml`)
Runs on every push and pull request to ensure code quality.

**Features:**
- Matrix testing across Python 3.8, 3.9, 3.10, and 3.11
- PostgreSQL database integration tests
- Code coverage reporting with Codecov
- Unit and integration test execution

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` branch

### 2. **Code Quality** (`code-quality.yml`)
Ensures code follows best practices and security standards.

**Features:**
- Code formatting checks with Black
- Import sorting with isort
- Linting with flake8 and pylint
- Type checking with mypy
- Security scanning with Bandit and Safety

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` branch

### 3. **Docker Build** (`docker-build.yml`)
Builds and publishes Docker images to GitHub Container Registry.

**Features:**
- Multi-stage Docker builds
- Automatic tagging based on branches and versions
- Cache optimization for faster builds
- Push to GitHub Container Registry (ghcr.io)

**Triggers:**
- Push to `main` branch
- Version tags (v*)
- Pull requests (build only, no push)

### 4. **Release** (`release.yml`)
Automates the release process when tags are created.

**Features:**
- Automatic changelog generation
- GitHub Release creation
- Python package building
- Optional PyPI publishing

**Triggers:**
- Push of version tags (v*)

### 5. **Dependabot** (`dependabot.yml`)
Keeps dependencies up to date automatically.

**Features:**
- Weekly Python dependency updates
- Monthly GitHub Actions updates
- Weekly npm dependency updates
- Automatic pull request creation

## 🔧 Required Secrets

Configure these secrets in your repository settings:

### Required:
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

### Optional:
- `PYPI_API_TOKEN` - For publishing to PyPI (if needed)
- `CODECOV_TOKEN` - For enhanced coverage reporting
- `DB_PASSWORD` - PostgreSQL password for production

## 📦 Initial Setup

### 1. Enable GitHub Actions
GitHub Actions should be automatically enabled. If not:
1. Go to Settings → Actions → General
2. Select "Allow all actions and reusable workflows"
3. Click Save

### 2. Configure Branch Protection
Protect your main branch:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - Require a pull request before merging
   - Require status checks to pass before merging
   - Select required checks: `test`, `lint`
   - Require branches to be up to date before merging

### 3. Set Up Environments (Optional)
Create deployment environments:
1. Go to Settings → Environments
2. Create `production` environment
3. Add protection rules and secrets as needed

## 🏃 Running Workflows Locally

Test workflows locally using `act`:

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash  # Linux

# Run all workflows
act

# Run specific workflow
act -W .github/workflows/python-tests.yml

# Run with specific event
act pull_request
```

## 📊 Status Badges

Add these badges to your README:

```markdown
![Tests](https://github.com/kevinmastascusa/GemCheck/workflows/Python%20Tests/badge.svg)
![Code Quality](https://github.com/kevinmastascusa/GemCheck/workflows/Code%20Quality/badge.svg)
![Docker](https://github.com/kevinmastascusa/GemCheck/workflows/Docker%20Build/badge.svg)
```

## 🐋 Docker Images

Published images are available at:
```
ghcr.io/kevinmastascusa/gemcheck:main
ghcr.io/kevinmastascusa/gemcheck:latest
ghcr.io/kevinmastascusa/gemcheck:v1.0.0
```

Pull the latest image:
```bash
docker pull ghcr.io/kevinmastascusa/gemcheck:latest
```

## 🔄 Workflow Triggers

| Workflow | Push (main) | Push (develop) | Pull Request | Tag (v*) | Schedule |
|----------|------------|----------------|--------------|----------|----------|
| Python Tests | ✅ | ✅ | ✅ | - | - |
| Code Quality | ✅ | ✅ | ✅ | - | - |
| Docker Build | ✅ | - | ✅ (no push) | ✅ | - |
| Release | - | - | - | ✅ | - |
| Dependabot | - | - | - | - | Weekly |

## 🛠️ Troubleshooting

### Tests Failing
1. Check the workflow logs in the Actions tab
2. Ensure all dependencies are in `requirements.txt`
3. Verify database migrations are up to date
4. Check for environment-specific issues

### Docker Build Issues
1. Verify Dockerfile syntax
2. Check for missing dependencies
3. Ensure build context is correct
4. Review multi-stage build steps

### Release Problems
1. Ensure tag follows semantic versioning (v1.0.0)
2. Check GITHUB_TOKEN permissions
3. Verify CHANGELOG generation
4. Review release asset paths

## 📚 Best Practices

1. **Commit Messages**: Use conventional commits for automatic changelog generation
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `chore:` Maintenance tasks

2. **Branch Strategy**:
   - `main` - Production-ready code
   - `develop` - Integration branch
   - `feature/*` - Feature branches
   - `hotfix/*` - Emergency fixes

3. **Version Tags**:
   - Use semantic versioning: `v1.0.0`
   - Create annotated tags: `git tag -a v1.0.0 -m "Release v1.0.0"`
   - Push tags: `git push origin v1.0.0`

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Dependabot Configuration](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)

---

**GitHub Actions are now configured and ready to use!** 🎉

Push to `main` or create a pull request to see the workflows in action.