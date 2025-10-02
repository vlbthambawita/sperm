# Security Guidelines

## 🔒 Sensitive Data Handling

This project has been organized to prevent accidental exposure of sensitive information like API keys and project IDs.

### What Was Removed/Protected

1. **Hardcoded API Keys**: All files previously containing hardcoded Labelbox API keys have been updated
2. **Project IDs**: Hardcoded project IDs replaced with environment variable references
3. **Sensitive Files**: Files containing secrets have been added to `.gitignore`

### Files That Were Removed

- `run.sh` - Contained hardcoded API key
- `labelbox_upload_checkpoint.json` - May contain sensitive data, should be generated locally

### Environment Variables Required

The following environment variables must be set:

```bash
LABELBOX_API_KEY=your_actual_api_key_here
PROJECT_ID=your_actual_project_id_here
DATA_FOLDER=/path/to/your/data
```

### Setup Instructions

1. **Copy the environment template**:
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` with your actual values**:
   ```bash
   # Edit .env file with your real API key and project ID
   nano .env
   ```

3. **Never commit `.env` to version control**:
   - The `.gitignore` file already excludes `.env`
   - Double-check before committing: `git status`

### Best Practices

- ✅ Use environment variables for all sensitive data
- ✅ Use `.env` files for local development
- ✅ Add all sensitive files to `.gitignore`
- ✅ Rotate API keys regularly
- ❌ Never hardcode secrets in source code
- ❌ Never commit `.env` files
- ❌ Never share API keys in chat/email

### Verification

Run the setup script to verify your configuration:
```bash
python scripts/setup.py
```

This will check:
- Environment variables are set
- Dependencies are installed  
- Connection to Labelbox works

## 🚨 If You Accidentally Commit Secrets

1. **Immediately rotate the exposed API key** in Labelbox
2. **Remove the secret from git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch path/to/file' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push to update remote**:
   ```bash
   git push origin --force --all
   ```
4. **Notify team members** to re-clone the repository
