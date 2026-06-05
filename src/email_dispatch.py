import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger("email_dispatch")

NEXT_STEPS_SUBJECT = "Next Steps to Open Your Account"
NEXT_STEPS_BODY_TEMPLATE = """\
Hi {first_name},

Thank you for starting your new account application with us. To finalize your
account opening, please have the following documents ready for verification:

  1. A government-issued photo ID (passport, driver's license, or national ID)
  2. Proof of residential address (utility bill or lease agreement from the
     last 90 days)
  3. Proof of employment or income (recent pay stub, employment letter, or
     tax return)
  4. Your identification number on file for cross-reference

A member of our verification team will reach out within one to two business
days to confirm receipt of these documents and complete your onboarding.

If you have any questions in the meantime, simply reply to this email.

Warm regards,
The Onboarding Team
"""


def send_next_steps_email(*, to_email: str, first_name: str) -> str:
    """Send the standardized 'Next Steps' onboarding email.

    Uses SMTP if SMTP_HOST is configured; otherwise logs the email and
    returns a 'logged' status so the workflow can continue in dev environments.
    """
    body = NEXT_STEPS_BODY_TEMPLATE.format(first_name=first_name)
    host = os.getenv("SMTP_HOST")
    if not host:
        logger.info(
            "SMTP not configured; would send next-steps email to %s\n%s",
            to_email,
            body,
        )
        return "logged"

    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", user or "noreply@example.com")

    msg = EmailMessage()
    msg["Subject"] = NEXT_STEPS_SUBJECT
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
    return "sent"
