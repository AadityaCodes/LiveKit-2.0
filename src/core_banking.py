import logging
import os
import secrets

from database import attach_account_numbers

logger = logging.getLogger("core_banking")

# ABC Bank's routing number is constant per real-world banking practice.
ROUTING_NUMBER = os.getenv("ABC_BANK_ROUTING_NUMBER", "021000021")


def _generate_account_number() -> str:
    # 10-digit checking account number, never starting with zero.
    first = secrets.choice("123456789")
    rest = "".join(secrets.choice("0123456789") for _ in range(9))
    return first + rest


def create_bank_account(customer_id: int) -> dict[str, str]:
    """Provision a new checking account for the given pending customer.

    Generates a unique account number, attaches it (along with the bank's
    routing number) to the pending_customers row, and returns the new
    identifiers.
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
