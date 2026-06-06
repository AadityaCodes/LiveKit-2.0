"""Mock Core Banking API.

In production this would call out to the real bank's account-opening
service. For development we generate a random 10-digit account number and
store it (plus a fixed routing number) onto the existing pending-customer
row. The interface stays identical so swapping in a real client later is a
local change.
"""

import logging
import os
import secrets

from database import attach_account_numbers

logger = logging.getLogger("core_banking")

# In real banking the routing number is fixed per institution. Treat it
# as a constant but allow override via env var for testing.
ROUTING_NUMBER = os.getenv("ABC_BANK_ROUTING_NUMBER", "021000021")


def _generate_account_number() -> str:
    """Return a random 10-digit account number (never starts with zero)."""
    # Bank account numbers conventionally don't start with 0.
    first = secrets.choice("123456789")
    rest = "".join(secrets.choice("0123456789") for _ in range(9))
    return first + rest


def create_bank_account(customer_id: int) -> dict[str, str]:
    """Provision a new checking account for the given pending customer.

    Generates a unique account number, attaches it (along with the bank's
    routing number) to the ``pending_customers`` row, and returns both
    values so the agent's Phase 5 tool can hand them to Phase 6.
    """
    account_number = _generate_account_number()
    attach_account_numbers(
        customer_id=customer_id,
        account_number=account_number,
        routing_number=ROUTING_NUMBER,
    )
    logger.info(
        "provisioned account %s (routing %s) for customer %d",
        account_number,
        ROUTING_NUMBER,
        customer_id,
    )
    return {"account_number": account_number, "routing_number": ROUTING_NUMBER}
