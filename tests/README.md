# GRPO Code Tests

This directory contains the test suite for the `grpo_code` package, migrated from the original test functions in `em_rewards_old.py`.

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Test only rewards functionality
pytest tests/test_rewards.py

# Test only transforms functionality  
pytest tests/test_transforms.py
```

### Run Integration Tests

Integration tests require the actual reward model to be available and may take longer to run:

```bash
# Run only integration tests
pytest -m integration

# Run all tests including integration tests
pytest -m "not integration or integration"
```

### Run Tests with Coverage

```bash
pip install pytest-cov
pytest --cov=grpo_code --cov-report=html
```

## Test Structure

- `test_rewards.py` - Tests for the Discord reaction prediction reward functions
- `test_transforms.py` - Tests for IRC conversation data transforms
- `conftest.py` - Shared fixtures and test configuration
- `README.md` - This documentation

## Test Categories

### Unit Tests
- Test individual functions with mocked dependencies
- Fast execution, no external dependencies required
- Run by default with `pytest`

### Integration Tests  
- Test with real models and dependencies
- Marked with `@pytest.mark.integration`
- May require model files to be available
- Run explicitly with `pytest -m integration`

## Original Test Migration

These tests were migrated from the original test functions:
- `test_reward_function()` → `test_rewards.py`
- `test_transform_function()` → `test_transforms.py`

The original test data and scenarios are preserved while adding proper pytest structure, mocking, and additional edge case coverage.
