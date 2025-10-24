# Contributing to Modela

Thank you for your interest in contributing to Modela! This document provides guidelines and information for contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/modela.git
   cd modela
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Test your changes**:
   - Upload a sample dataset (Iris, Titanic, etc.)
   - Train a model
   - Verify all functionality works

## 📝 How to Contribute

### Bug Reports
- Use the GitHub issue tracker
- Include steps to reproduce
- Provide error messages and system information

### Feature Requests
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider if it aligns with the project's goals

### Code Contributions
- Follow the existing code style
- Add comments for complex logic
- Write clear commit messages
- Test your changes thoroughly

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization**: Improve training speed
- **Additional Algorithms**: Add more ML models to PyCaret
- **Data Preprocessing**: Enhanced data cleaning features
- **Visualization**: New chart types and insights

### Medium Priority
- **API Integration**: REST API endpoints
- **Model Deployment**: Export to various formats
- **Advanced Features**: Hyperparameter tuning
- **Documentation**: Tutorials and examples

### Low Priority
- **UI/UX Improvements**: Better user interface
- **Testing**: Unit tests and integration tests
- **Internationalization**: Multi-language support
- **Mobile App**: React Native or Flutter app

## 📋 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update CHANGELOG.md** with your changes
5. **Submit a pull request** with a clear description

### Pull Request Template
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
- [ ] All existing tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## 🏗️ Code Style Guidelines

### Python
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions small and focused

### Streamlit
- Use consistent naming conventions
- Organize code into logical sections
- Add helpful comments for complex UI logic
- Use session state appropriately

## 🐛 Testing

### Manual Testing Checklist
- [ ] Upload different file formats (CSV, Excel)
- [ ] Test with various dataset sizes
- [ ] Verify model training works
- [ ] Check visualization accuracy
- [ ] Test download functionality
- [ ] Verify user authentication

### Automated Testing
```bash
# Run linting
flake8 .

# Run type checking
mypy .

# Run tests (when implemented)
pytest
```

## 📚 Documentation

- Update README.md for new features
- Add docstrings to new functions
- Include examples in documentation
- Update deployment instructions if needed

## 🎉 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Given credit in the project

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and features
- **Email**: Contact the maintainer directly

## 📄 License

By contributing to Modela, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Modela! 🚀
