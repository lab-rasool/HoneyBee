# Contributing to HoneyBee

Thank you for your interest in contributing to HoneyBee! We welcome contributions from the community and are grateful for any help you can provide. This guide will help you get started with contributing to our multimodal AI framework for oncology.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Contribution Guidelines](#contribution-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be Respectful**: Treat everyone with respect. No harassment, discrimination, or inappropriate behavior will be tolerated.
- **Be Collaborative**: Work together to resolve conflicts and assume good intentions.
- **Be Professional**: Keep discussions focused on improving the project.
- **Be Inclusive**: Welcome newcomers and help them get started.

## Getting Started

1. **Fork the Repository**: Click the "Fork" button on the [HoneyBee GitHub page](https://github.com/lab-rasool/HoneyBee)

2. **Star the Repository**: If you find HoneyBee useful, please star it to show your support!

3. **Check Issues**: Look through our [open issues](https://github.com/lab-rasool/HoneyBee/issues) for something to work on:
   - Issues labeled `good first issue` are great for newcomers
   - Issues labeled `help wanted` need attention
   - Issues labeled `enhancement` are for new features

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git
- CUDA 11.7+ (optional, for GPU support)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y openslide-tools tesseract-ocr

# macOS
brew install openslide tesseract
```

### Setting Up Your Development Environment

1. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HoneyBee.git
   cd HoneyBee
   ```

2. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/lab-rasool/HoneyBee.git
   ```

3. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   python -c "import nltk; nltk.download('punkt')"
   ```

5. **Install HoneyBee in Development Mode**
   ```bash
   pip install -e .
   ```

6. **Set Up Environment Variables**
   Create a `.env` file in the project root (see README.md for details)

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check if the issue already exists. If not, create a new issue with:

- **Clear title**: Summarize the bug concisely
- **Description**: Detailed description of the bug
- **Steps to reproduce**: List the exact steps to reproduce the behavior
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment details**: Python version, OS, GPU info, etc.
- **Error messages**: Include full error traceback
- **Code samples**: Minimal code to reproduce the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title**: Summarize the enhancement
- **Motivation**: Why is this enhancement needed?
- **Detailed description**: How should it work?
- **Alternative solutions**: Any alternative solutions you've considered
- **Additional context**: Mockups, diagrams, or examples

### Contributing Code

1. **Find or Create an Issue**: Before starting work, ensure there's an issue for what you're working on

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Your Changes**
   - Follow our coding standards (see below)
   - Add or update tests as needed
   - Update documentation if necessary

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature X"
   # or
   git commit -m "fix: resolve issue with Y"
   ```

   We follow conventional commits:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting, etc.)
   - `refactor:` Code refactoring
   - `test:` Test additions or modifications
   - `chore:` Maintenance tasks

5. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**: Go to GitHub and create a PR from your fork to the main repository

## Contribution Guidelines

### Code Style

- **Python Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type Hints**: Use type hints where appropriate
- **Line Length**: Maximum 100 characters
- **Imports**: Sort imports with `isort`

Example:
```python
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from honeybee.models.base import BaseModel


class NewModel(BaseModel):
    """Brief description of the model.
    
    Longer description explaining the model's purpose,
    architecture, and use cases.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output
        
    Example:
        >>> model = NewModel(input_dim=512, hidden_dim=256, output_dim=10)
        >>> embeddings = model.generate_embeddings(data)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def generate_embeddings(
        self,
        data: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings for input data.
        
        Args:
            data: Input tensor
            batch_size: Optional batch size for processing
            
        Returns:
            Embeddings as numpy array
        """
        # Implementation here
        pass
```

### Architecture Guidelines

When adding new components, follow the 3-layer architecture:

1. **Data Loaders** (`honeybee/loaders/`)
   - Inherit from appropriate base class
   - Implement standard loading interface
   - Handle error cases gracefully
   - Document supported formats

2. **Embedding Models** (`honeybee/models/`)
   - Implement `generate_embeddings()` method
   - Support both CPU and GPU
   - Include model weights handling
   - Document model requirements

3. **Processors** (`honeybee/processors/`)
   - Combine loaders and models
   - Implement preprocessing pipelines
   - Handle multimodal integration
   - Provide clear configuration options

### Adding New Features

1. **Discuss First**: For major features, open an issue for discussion before implementation
2. **Backward Compatibility**: Ensure changes don't break existing functionality
3. **Configuration**: Make new features configurable when possible
4. **Examples**: Add example usage in `examples/` directory
5. **Documentation**: Update relevant documentation

## Testing

Currently, HoneyBee uses example scripts for testing. When contributing:

1. **Test Your Changes**
   ```bash
   # Run relevant example scripts
   python clinical/test_clinical_processing.py
   python examples/survival.py
   ```

2. **Add Test Cases**: If adding new functionality, create corresponding test scripts

3. **Test Multiple Environments**: Test on different Python versions and with/without GPU

Future: We plan to implement a comprehensive test suite using pytest.

## Documentation

### Code Documentation

- All public functions and classes must have docstrings
- Use clear, descriptive variable names
- Add inline comments for complex logic
- Include type hints

### User Documentation

When adding new features:

1. Update the main `README.md` if needed
2. Update `CLAUDE.md` with relevant information
3. Add example notebooks to `examples/`
4. Update the website documentation in `website/`

### Website Development

The documentation website uses Astro:

```bash
cd website
npm install
npm run dev  # Start development server
```

## Pull Request Process

1. **Update Your Branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure Quality**
   - [ ] Code follows style guidelines
   - [ ] Tests pass (run example scripts)
   - [ ] Documentation is updated
   - [ ] Commit messages follow conventions
   - [ ] No sensitive data (keys, passwords) included

3. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tested locally
   - [ ] Added new tests
   - [ ] All tests pass
   
   ## Checklist
   - [ ] My code follows the project style guidelines
   - [ ] I have performed a self-review
   - [ ] I have commented my code where necessary
   - [ ] I have updated the documentation
   - [ ] My changes generate no new warnings
   ```

4. **Review Process**
   - A maintainer will review your PR
   - Address any requested changes
   - Once approved, your PR will be merged

## Community

### Getting Help

- **Issues**: Use GitHub issues for bugs and features
- **Discussions**: Use GitHub Discussions for questions and ideas

### Recognition

Contributors are recognized in several ways:
- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Co-authorship on related publications (for substantial research contributions)

## Thank You!

Your contributions make HoneyBee better for everyone in the oncology AI research community. We appreciate your time and effort in improving this project!

---

If you have any questions about contributing, please don't hesitate to ask. We're here to help and look forward to your contributions! 🐝