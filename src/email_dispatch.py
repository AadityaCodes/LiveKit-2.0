"""Welcome-email dispatch.

Composes and sends the Phase 6 "Welcome / Next Steps" email that contains
the newly provisioned Account ID, routing number, login URL, and a fresh
temporary password.

If ``SMTP_HOST`` is configured the message is sent via SMTP; otherwise
the email body is logged so the local dev/eval flow can complete without
real email infrastructure.

Environment overrides:
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / SMTP_FROM
  ABC_BANK_LOGIN_URL — link shown in the welcome email body
"""

import logging
import os
import secrets
import smtplib
import string
from email.message import EmailMessage

logger = logging.getLogger("email_dispatch")

WELCOME_SUBJECT = "Your new ABC Bank checking account is open"
LOGIN_URL = os.getenv("ABC_BANK_LOGIN_URL", "https://online.abcbank.example.com/login")
WELCOME_BODY_TEMPLATE = """\
Hi {first_name},

Welcome to ABC Bank! Your new checking account is officially open. Here are
your account details:

  Account ID (Account Number): {account_number}
  Routing Number: {routing_number}

To activate your online banking, sign in at:

  {login_url}

Use these temporary credentials on your first sign-in (you will be prompted
to set a permanent password):

  Username: {account_number}
  Temporary password: {temp_password}

A welcome packet with your debit card and checks will arrive at your
residential address within 7 to 10 business days.

If you have any questions, simply reply to this email or call us back.

Warm regards,
The ABC Bank Onboarding Team
"""


def _generate_temp_password(length: int = 12) -> str:
    """Return a cryptographically random alphanumeric temporary password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def send_welcome_email(
    *,
    to_email: str,
    first_name: str,
    account_number: str,
    routing_number: str,
) -> dict[str, str]:
    """Send the welcome email with newly provisioned account details.

    Generates a dummy temporary password, includes a login URL and the
    account ID, and either sends via SMTP (if SMTP_HOST is set) or logs the
    email body for dev environments.

    Returns a dict with 'status' ('sent' or 'logged') and 'temp_password'.
    """
    temp_password = _generate_temp_password()
    body = WELCOME_BODY_TEMPLATE.format(
        first_name=first_name,
        account_number=account_number,
        routing_number=routing_number,
        login_url=LOGIN_URL,
        temp_password=temp_password,
    )
    host = os.getenv("SMTP_HOST")
    if not host:
        logger.info(
            "SMTP not configured; would send welcome email to %s\n%s",
            to_email,
            body,
        )
        return {"status": "logged", "temp_password": temp_password}

    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", user or "noreply@abcbank.example.com")

    msg = EmailMessage()
    msg["Subject"] = WELCOME_SUBJECT
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return {"status": "sent", "temp_password": temp_password}
