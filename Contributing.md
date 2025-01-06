# Contributing to piQture

Thank you for your interest in contributing to piQture! We're excited to welcome contributions from quantum computing enthusiasts, machine learning practitioners, and developers. This document provides guidelines for contributing to the project.

## Code of Conduct

Our community strives to be open, inclusive, and respectful. We expect all contributors to adhere to our Code of Conduct in all project interactions.

## Getting Started

### Development Environment Setup

1. Create and activate a Python environment:
```bash
# Using conda
conda create -n piqture_dev python=3.x
conda activate piqture_dev

# OR using venv
python -m venv piqture_dev
source piqture_dev/bin/activate  # Unix
.\piqture_dev\Scripts\activate   # Windows
```

2. Clone and install development dependencies:
```bash
git clone https://github.com/SaashaJoshi/piQture.git
cd piQture
pip install -r requirements.txt
pip install -e .
```

### Running Tests

We use pytest for our test suite:
```bash
tox
```

## How to Contribute

### Areas We Need Help With

- Implementing new quantum image encoding methods
- Optimizing quantum circuit implementations
- Adding support for new quantum machine learning models
- Improving documentation and adding examples
- Writing tests and improving test coverage
- Performance optimization and benchmarking
- And other good first issues
### Pull Request Process

1. Fork the repository and create a new branch:
```bash
git checkout -b branch-name
```

2. Make your changes following our code style guidelines
3. Write or update tests as needed
4. Update documentation if you're introducing new features
5. Run the test suite to ensure everything passes
6. Push your changes and create a pull request

#### Pull Request Guidelines

- Use a clear, descriptive title
- Include the purpose of the change and relevant issue numbers
- Provide before/after examples for visual changes
- Include performance benchmarks for optimization changes
- Update docstrings and comments as needed

### Commit Message Format

```
type(scope): Brief description

Longer description if needed

Closes #123
```

Types (Optional):
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

## Code Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Maximum line length: 88 characters (using Black formatter)
- Use docstrings for classes and functions
Code formatting can be checked using `tox`.
 ```bash
  tox -e lint
### Quantum Circuit Style

- Comment complex quantum operations
- Include circuit diagrams in docstrings when helpful
- Optimize circuit depth where possible

Example:
```python
def apply_quantum_convolution(
    circuit: QuantumCircuit,
    qreg: QuantumRegister,
    params: Dict[str, Any]
) -> None:
    """Apply quantum convolution operation to the circuit.

    Args:
        circuit: The quantum circuit to modify
        qreg: Quantum register for the operation
        params: Dictionary containing convolution parameters

    Returns:
        None
    """
    # Implementation
```

## Documentation

- Keep docstrings up to date
- Add examples for new features
- Update README.md when adding major features
- Include references to relevant papers or resources

## Testing

- Write unit tests for new features
- Include test cases for edge cases
- Add integration tests for complex workflows
- Maintain or improve test coverage

## Issue Guidelines

When creating issues:

1. Check existing issues to avoid duplicates
2. Use issue templates when available
3. Include:
   - Python/piQture version
   - Operating system
   - Clear steps to reproduce
   - Expected vs actual behavior
   - Relevant error messages
   - Code samples if applicable

## Getting Help

- Open an issue for bugs or feature requests
- Join our community discussions
- Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Acknowledgments

We appreciate all contributors who help make piQture better! Your contributions, whether big or small, are valuable to the project's success.
