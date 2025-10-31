# Contributing to Clash Emote Detector

Thank you for your interest in contributing to the Clash Royale Emote Detector! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior vs actual behavior
- Your environment (OS, Python version, etc.)
- Screenshots or error logs (if applicable)

### Suggesting Features

We love new ideas! To suggest a feature:
1. Check if it's already been suggested in Issues
2. Open a new issue with the label "enhancement"
3. Describe the feature and why it would be useful
4. Provide examples or mockups if possible

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add docstrings to new functions
   - Keep commits focused and atomic
   - Write clear commit messages

3. **Test your changes**:
   - Ensure the app runs without errors
   - Test detection with multiple emotes
   - Check that camera initialization/release works properly
   - Verify no regression in existing features

4. **Update documentation**:
   - Update README.md if you add new features
   - Add comments to complex code sections
   - Update CHANGELOG.md with your changes

5. **Submit the PR**:
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots/videos for UI changes

## 📝 Code Style Guidelines

### Python
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines when possible
- Use type hints where appropriate

Example:
```python
def detect_emote(frame: np.ndarray, confidence_threshold: float = 0.45) -> tuple[str, float]:
    """
    Detect emote from a video frame.
    
    Args:
        frame: Input image as numpy array (BGR format)
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        tuple: (emote_name, confidence_score)
    """
    # Implementation here
    pass
```

### JavaScript
- Use ES6+ syntax
- Add comments for complex logic
- Keep functions small and reusable
- Use meaningful variable names

### HTML/CSS
- Use semantic HTML5 tags
- Follow TailwindCSS conventions
- Keep styles consistent with existing design
- Ensure responsive design

## 🔧 Development Setup

1. Fork and clone the repository
2. Run `setup.bat` to install dependencies
3. Create a new branch for your feature
4. Make changes and test thoroughly
5. Commit and push to your fork
6. Open a pull request

## 🧪 Testing

Before submitting a PR, please test:

### Manual Testing Checklist
- [ ] Camera initializes properly
- [ ] All pages load without errors
- [ ] Emote detection works with good accuracy
- [ ] Audio plays correctly on detection
- [ ] Model switching works without restart
- [ ] Settings persist after page reload
- [ ] Data collection captures frames properly
- [ ] Manage emotes page functions correctly

### Performance Testing
- [ ] FPS remains above 15 on average hardware
- [ ] Memory usage is reasonable (<1GB)
- [ ] No memory leaks during extended use
- [ ] Camera releases properly when switching pages

## 📦 Project Areas

### Easy Contributions (Good First Issues)
- Documentation improvements
- Bug fixes
- UI/UX enhancements
- Adding new emotes
- Improving error messages

### Intermediate Contributions
- New feature development
- Performance optimizations
- Adding tests
- Refactoring code
- Improving model architecture

### Advanced Contributions
- Multi-person detection
- Real-time audio processing
- Cloud integration
- Mobile app development
- Advanced ML models

## 🎯 Priority Areas

We're especially interested in contributions for:
1. **Better model architectures** - Improve detection accuracy
2. **Cross-platform support** - Make it work on Linux/Mac
3. **Mobile app** - React Native or Flutter version
4. **Testing framework** - Add automated tests
5. **Documentation** - More tutorials and guides

## 🚫 What Not to Include

Please avoid:
- Large binary files (models >10MB should be hosted externally)
- Proprietary/copyrighted content
- Breaking changes without discussion
- Unrelated dependencies
- Code without documentation

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inspiring community for everyone.

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize what's best for the community

### Unacceptable Behavior
- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Unprofessional conduct

## 📧 Questions?

If you have questions about contributing:
- Open a discussion on GitHub
- Email: parth.ajit7052@gmail.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Clash Emote Detector!** 🎉
