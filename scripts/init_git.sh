#!/bin/bash

# Initialize Git Repository for Sperm Labelbox Uploader
# This script sets up a clean git repository ready for GitHub

echo "🚀 Initializing Git repository..."

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
git add .

# Check what's being added (should not include sensitive files)
echo ""
echo "📋 Files to be committed:"
git status --porcelain

# Check if .env or other sensitive files are accidentally staged
if git status --porcelain | grep -E "\.(env|key|secret)" > /dev/null; then
    echo "⚠️  WARNING: Sensitive files detected in staging area!"
    echo "Please review and remove them before committing."
    exit 1
fi

# Initial commit
git commit -m "Initial commit: Sperm detection Labelbox uploader

- Organized project structure with src/, examples/, docs/, scripts/
- Removed hardcoded API keys and sensitive data
- Added environment variable configuration
- Created comprehensive documentation
- Added security guidelines and setup scripts"

echo ""
echo "✅ Initial commit created successfully!"
echo ""
echo "📚 Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add the remote: git remote add origin <your-repo-url>"
echo "3. Push to GitHub: git push -u origin main"
echo ""
echo "🔒 Security reminder:"
echo "- Never commit your .env file"
echo "- Keep your API keys secure"
echo "- Review files before each commit"
