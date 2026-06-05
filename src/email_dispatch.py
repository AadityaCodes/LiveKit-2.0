import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger("email_dispatch")

WELCOME_SUBJECT = "Your new ABC Bank checking account is open"
WELCOME_BODY_TEMPLATE = """\
Hi {first_name},

Welcome to ABC Bank! Your new checking account is officially open. Here are
your account details:

  Account Number: {account_number}
  Routing Number: {routing_number}

Next steps to set up your online banking:

  1. Visit abcbank.example.com and click "Enroll in Online Banking".
  2. Verify your identity using the account number above and the last four
     digits of your identification number on file.
  3. Create a username and password, then set up multi-factor authentication.
  4. Download the ABC Bank mobile app from your app store and sign in with
     your new credentials.

A welcome packet with your debit card and checks will arrive at your
residential address within 7 to 10 business days.

If you have any questions, simply reply to this email or call us back.

Warm regards,
The ABC Bank Onboarding Team
"""


def send_welcome_email(
    *,
    to_email: str,
    first_name: str,
    account_number: str,
    routing_number: str,
) -> str:
    """Send the welcome email with newly provisioned account details.

    Uses SMTP if SMTP_HOST is configured; otherwise logs the email and
    returns a 'logged' status so the workflow can complete in dev.
    """
    body = WELCOME_BODY_TEMPLATE.format(
        first_name=first_name,
        account_number=account_number,
        routing_number=routing_number,
    )
    host = os.getenv("SMTP_HOST")
    if not host:
        logger.info(
            "SMTP not configured; would send welcome email to %s\n%s",
            to_email,
            body,
        )
        return "logged"

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
    return "sent"
