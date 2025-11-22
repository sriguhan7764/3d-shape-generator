# Contributing to 3D Generative Design for Additive Manufacturing

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/sriguhan7764/3d-shape-generator/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU if applicable)
   - Error messages or logs

### Suggesting Enhancements

1. Check existing issues and discussions
2. Create a new issue describing:
   - The feature or enhancement
   - Use cases and benefits
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes:
   - Write clean, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed
4. Test your changes thoroughly
5. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request with:
   - Clear description of changes
   - Related issue numbers
   - Screenshots (if UI changes)

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/sriguhan7764/3d-shape-generator.git
   cd 3d-shape-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p checkpoints data
   ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and concise
- Comment complex algorithms

## Testing

Before submitting a PR:

- Test the web interface locally
- Verify training script works (if modified)
- Check export functionality (OBJ/STL)
- Test on different resolutions if changing model architecture

## Areas for Contribution

We especially welcome contributions in:

- **Model Improvements**: New architectures, training techniques
- **UI/UX Enhancements**: Better visualization, user experience
- **Performance Optimization**: Faster generation, lower memory usage
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Unit tests, integration tests
- **Dataset Support**: Additional datasets, data augmentation
- **Export Formats**: Support for more 3D file formats

## Questions?

Feel free to:
- Open an issue for clarification
- Start a discussion
- Contact the maintainers

Thank you for contributing!
