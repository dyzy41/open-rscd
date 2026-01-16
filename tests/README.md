# Tests

This directory contains unit tests for the ExchangeNet implementation.

## Running Tests

Run all tests:
```bash
# Test exchange modules
python tests/test_exchange.py

# Test model architecture
python tests/test_model.py

# Test utility functions
python tests/test_utils.py
```

## Test Coverage

- **test_exchange.py**: Tests for Exchange modules (ExchangeBlock, SpatialExchange, ExchangeModule)
- **test_model.py**: Tests for the main ExchangeNet model and build functions
- **test_utils.py**: Tests for loss functions, metrics, and utilities

## Adding New Tests

When adding new functionality, please add corresponding tests following the existing patterns:

1. Import necessary modules
2. Create test functions that check expected behavior
3. Use assertions to verify correctness
4. Run tests to ensure they pass
