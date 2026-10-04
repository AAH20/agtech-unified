# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes    |
| < 1.0   | ❌ No     |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Do NOT open a public issue.**

Instead, report via one of these channels:

1. **GitHub Security Advisories** (preferred):
   - Go to the [Security Advisories](https://github.com/AAH20/agtech-unified/security/advisories) page
   - Click "Report a vulnerability"
   - Provide detailed information

2. **Email**: security@agtech-unified.dev

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Affected versions

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 5 business days
- **Fix or mitigation**: Within 30 days (for critical issues)

## Security Features

This project implements several security measures:

- **Zero-trust authentication** via JWT tokens
- **Role-based access control** (RBAC)
- **Rate limiting** on API endpoints
- **GPS anti-spoofing** detection for IoT devices
- **Audit logging** for compliance
- **Password policy** enforcement
- **TOTP two-factor authentication** support

## Scope

### In Scope

- Authentication and authorization bypass
- Data exfiltration vulnerabilities
- Injection attacks (SQL, command, etc.)
- Cross-site scripting (XSS)
- Cross-site request forgery (CSRF)
- Insecure deserialization
- Sensitive data exposure

### Out of Scope

- Denial of service (DoS) attacks
- Social engineering
- Physical security
- Third-party dependency vulnerabilities (report to upstream)

## Security Best Practices for Users

1. **Never commit secrets** — use `.env` files (see `.env.example`)
2. **Rotate JWT secrets** regularly
3. **Enable TLS** in production
4. **Use strong passwords** — the default policy requires 12+ characters
5. **Enable 2FA** for admin accounts
6. **Monitor audit logs** for suspicious activity
7. **Keep dependencies updated** — run `pip audit` regularly
