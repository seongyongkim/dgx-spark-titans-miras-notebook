# Contributing to Titans+MIRAS Notebooks

Thank you for your interest in contributing to this educational project! This document provides guidelines for contributing.

## 🎯 Ways to Contribute

### 1. Report Issues
- **Bug Reports**: If a notebook cell fails or produces unexpected results
- **Documentation**: Typos, unclear explanations, or missing information
- **Feature Requests**: Ideas for new visualizations or educational content

### 2. Submit Pull Requests
- Fix bugs or typos
- Improve explanations or add comments
- Add new visualizations or examples
- Optimize code for better performance

## 📋 Guidelines

### For Code Changes

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Test your changes** by running the entire notebook from top to bottom

3. **Clear notebook outputs** before committing:
   - In Jupyter: `Kernel > Restart & Clear Output`
   - Or use: `jupyter nbconvert --clear-output --inplace notebook.ipynb`

4. **Follow the existing code style**:
   - Use clear, descriptive variable names
   - Add comments explaining "why" not just "what"
   - Keep visualizations beginner-friendly

5. **Submit a Pull Request** with:
   - Clear description of changes
   - Screenshots if adding visualizations
   - Reference to any related issues

### For Documentation

- Keep explanations accessible to beginners
- Use analogies and real-world examples
- Include code comments for complex operations

## 🧪 Testing Checklist

Before submitting, ensure:

- [ ] All cells run without errors (top to bottom)
- [ ] Visualizations render correctly
- [ ] No hardcoded file paths
- [ ] Works on Google Colab
- [ ] Requirements.txt is updated if new dependencies added

## 💬 Questions?

Feel free to open an issue for any questions about contributing.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
