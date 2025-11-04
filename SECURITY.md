# Security Policy

## Overview

VantageAdapt takes security seriously. This document outlines security best practices and how to report vulnerabilities.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

### API Keys and Secrets

**NEVER** commit API keys, secrets, or credentials to version control:

1. **Use Environment Variables**: All sensitive data must be stored in environment variables
   ```bash
   # Good ✓
   api_key = os.getenv("OPENAI_API_KEY")

   # Bad ✗
   api_key = "sk-proj-..."
   ```

2. **Use .env Files Locally**: Copy `.env.example` to `.env` and add your keys
   ```bash
   cp .env.example .env
   # Edit .env with your actual keys
   ```

3. **Never Commit .env**: The `.gitignore` file already excludes `.env`, but be careful with new files

4. **Rotate Compromised Keys**: If you accidentally commit a key:
   - Immediately revoke it from the provider's dashboard
   - Generate a new key
   - Update your `.env` file
   - Consider using `git-secrets` or similar tools

### Required Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY` - OpenAI API key for LLM operations
- `LANGCHAIN_API_KEY` - LangSmith API key for tracing (optional but recommended)
- `MEM0_API_KEY` - Mem0 API key for memory management
- `DATABASE_URL` - PostgreSQL database connection string

See `.env.example` for a complete list with descriptions.

### Database Security

1. **Use Strong Passwords**: Database passwords should be:
   - At least 16 characters long
   - Include uppercase, lowercase, numbers, and special characters
   - Never use default passwords

2. **Restrict Database Access**:
   - Only allow connections from trusted IP addresses
   - Use SSL/TLS for database connections in production
   - Never expose your database directly to the internet

3. **Backup Encryption**: Ensure database backups are encrypted at rest

### Code Security

1. **Input Validation**: Always validate and sanitize user inputs
2. **SQL Injection Prevention**: Use parameterized queries (SQLAlchemy handles this)
3. **Dependency Management**: Keep dependencies up to date
   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Code Review**: All PRs should be reviewed for security issues

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **DO NOT** open a public GitHub issue
2. Email the maintainers directly (contact info in README)
3. Provide detailed information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

We will:
- Acknowledge receipt within 48 hours
- Provide a detailed response within 7 days
- Work on a fix and coordinate disclosure timing with you

## Security Checklist for Contributors

Before submitting a PR, ensure:

- [ ] No API keys or secrets in code
- [ ] No hardcoded credentials
- [ ] Environment variables used for all sensitive data
- [ ] Dependencies are up to date
- [ ] Input validation is implemented
- [ ] Error messages don't leak sensitive information
- [ ] Tests don't contain real API keys (use mocks/fixtures)

## Security Tools

We recommend using these tools:

- **git-secrets**: Prevent committing secrets
  ```bash
  git secrets --install
  git secrets --register-aws
  ```

- **bandit**: Python security linter
  ```bash
  pip install bandit
  bandit -r .
  ```

- **safety**: Check for known vulnerabilities
  ```bash
  pip install safety
  safety check
  ```

## Acknowledgments

We thank the security community for responsibly disclosing vulnerabilities.
