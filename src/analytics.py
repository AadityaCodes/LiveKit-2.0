"""Post-call customer-insights analytics.

After a call closes we feed the JSONL transcript to Gemini with a
strictly-structured prompt and extract a JSON insights blob describing
the customer: sentiment, life events they mentioned, products they
asked about, preferred channels, and concrete marketing-campaign ideas.

Output goes two places:

* ``customer_insights`` table in SQLite, keyed by ``account_number``
  (upserted, so the most recent call wins). Downstream marketing jobs
  query this table.
* ``analytics/<transcript_stem>.json`` on disk — human-readable archive
  of the same payload alongside the transcript and eval report.

We rely on the provisioned ``account_number`` as the primary key. If the
call never reached Phase 5 (no provisioning happened) we skip the run —
there's no stable key to write under.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from database import upsert_customer_insights

logger = logging.getLogger("analytics")

# Where the per-call JSON archives land.
ANALYTICS_DIR = Path("analytics")

# Same model family as the eval; Flash is fine for structured extraction.
_ANALYTICS_MODEL = os.getenv("GEMINI_ANALYTICS_MODEL", "gemini-flash-latest")

# Env var that holds the Gemini API key (loaded from .env.local at startup).
_GEMINI_KEY_ENV = "GOOGLE_GEMINI_API_KEY"

# The schema we ask Gemini to produce. Keeping it in one place keeps the
# DB consumers and the prompt aligned.
_INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Two- or three-sentence neutral summary of the call from a "
                "relationship-banking perspective."
            ),
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
            "description": "Caller's overall sentiment toward ABC Bank.",
        },
        "personal_interests": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Hobbies, lifestyle, or interests the caller mentioned in "
                "passing (e.g. 'travel', 'home renovation', 'cycling')."
            ),
        },
        "life_events": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Major life events implied or stated: 'newly married', "
                "'expecting a child', 'recently relocated', 'started new "
                "job', 'planning retirement', etc."
            ),
        },
        "financial_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Hints about the caller's financial situation or goals: "
                "'saving for a house', 'paying off student loans', "
                "'frequent international transfers'."
            ),
        },
        "product_interest": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "ABC Bank products the caller showed any curiosity about "
                "beyond the checking account (e.g. 'credit card', "
                "'mortgage', 'investment account')."
            ),
        },
        "preferred_contact_method": {
            "type": "string",
            "description": (
                "What the caller said about how to reach them, or 'unknown' "
                "if not mentioned."
            ),
        },
        "marketing_recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "2-5 concrete, sensitive, compliant follow-up campaign "
                "ideas the bank could send to this customer."
            ),
        },
        "do_not_contact_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Anything the caller said that suggests they would NOT "
                "welcome marketing on a topic. Empty array if none."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Free-form notes worth surfacing to a human banker.",
        },
    },
    "required": [
        "summary",
        "sentiment",
        "personal_interests",
        "life_events",
        "financial_signals",
        "product_interest",
        "preferred_contact_method",
        "marketing_recommendations",
        "do_not_contact_signals",
        "notes",
    ],
}

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a senior CRM analyst at ABC Bank. You are reading the JSONL
    transcript of a phone call between a customer and an autonomous voice
    agent that opens new checking accounts.

    Your job is to extract a single, structured JSON record describing the
    customer for use in future personalized service and marketing.

    Rules:
      * Only include facts that are actually present in the transcript.
        Do not invent details. If something is unknown, leave the field
        empty (empty string or empty array as appropriate). For
        ``preferred_contact_method``, use the literal string "unknown".
      * Be specific. "Saving for a house" is useful; "saving money" is not.
      * Do NOT include raw PII (identification number, full date of birth,
        full home address). High-level signals are fine ("recently moved",
        "lives in metro area").
      * Marketing recommendations must be respectful, compliant, and tied
        to something the customer actually said. Never suggest predatory
        products, never recommend anything based on protected
        characteristics, and respect any do-not-contact signals.
      * Output MUST be a single JSON object that conforms to the provided
        schema. No prose, no Markdown, no code fences.
    """
)


# Two regexes used to recover identifiers from the transcript without
# re-parsing JSON streams. They look at the OK payload string the tools
# return.
_RE_ACCOUNT = re.compile(r"account_number=(\d+)")
_RE_CUSTOMER_ID = re.compile(r"customer_id=(\d+)")


def _scan_transcript(path: Path) -> tuple[str, str | None, int | None]:
    """Return ``(transcript_text, account_number, customer_id)``.

    ``transcript_text`` is normalized JSONL (one event per line). The two
    identifiers are pulled out of the tool-call outputs if Phase 4 / Phase 5
    completed during this call.
    """
    if not path.exists():
        return "", None, None

    normalized: list[str] = []
    account_number: str | None = None
    customer_id: int | None = None
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
                normalized.append(json.dumps(event, ensure_ascii=False))
            except json.JSONDecodeError:
                normalized.append(raw)
                continue

            if event.get("event") != "tool_call":
                continue
            output = event.get("output") or ""
            if event.get("name") == "provision_bank_account":
                m = _RE_ACCOUNT.search(output)
                if m:
                    account_number = m.group(1)
            elif event.get("name") == "collect_customer_information":
                m = _RE_CUSTOMER_ID.search(output)
                if m:
                    with contextlib.suppress(ValueError):
                        customer_id = int(m.group(1))

    return "\n".join(normalized), account_number, customer_id


def _parse_json_payload(raw: str) -> dict[str, Any] | None:
    """Robustly parse the model output as JSON.

    The model is asked for raw JSON. If it slips and wraps the payload in
    ```json ... ``` fences anyway, strip those and retry.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (with optional 'json' tag) and the
        # closing fence.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        logger.warning("could not parse analytics JSON payload: %r", raw[:300])
        return None


async def run_post_call_analytics(transcript_path: Path) -> Path | None:
    """Read the call's transcript, extract customer insights, persist them.

    Returns the path of the on-disk JSON archive on success, or ``None``
    if the run was skipped (no account number, no API key, parse error,
    etc.). Never raises — all failures are logged.
    """
    transcript, account_number, customer_id = _scan_transcript(transcript_path)
    if not transcript:
        logger.warning("analytics skipped: empty transcript at %s", transcript_path)
        return None
    if not account_number:
        logger.info(
            "analytics skipped: no account_number provisioned during call %s",
            transcript_path,
        )
        return None

    api_key = os.getenv(_GEMINI_KEY_ENV)
    if not api_key:
        logger.warning(
            "analytics skipped: %s is not set; cannot reach Gemini",
            _GEMINI_KEY_ENV,
        )
        return None

    client = genai.Client(api_key=api_key)
    user_prompt = (
        "JSONL transcript follows. Extract the structured insights JSON now.\n\n"
        f"{transcript}"
    )

    try:
        response = await client.aio.models.generate_content(
            model=_ANALYTICS_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
                # Asking for structured output keeps the model on schema.
                response_mime_type="application/json",
                response_schema=_INSIGHTS_SCHEMA,
            ),
        )
    except Exception:
        logger.exception(
            "Gemini analytics call failed for transcript %s", transcript_path
        )
        return None

    payload = _parse_json_payload(response.text or "")
    if payload is None:
        return None

    # Persist to SQLite for downstream marketing jobs / future calls.
    try:
        upsert_customer_insights(
            account_number=account_number,
            customer_id=customer_id,
            summary=payload.get("summary", ""),
            sentiment=payload.get("sentiment", "neutral"),
            insights=payload,
            source_transcript=str(transcript_path),
        )
    except Exception:
        logger.exception("failed to upsert customer_insights row")
        # Continue to write the on-disk archive even if SQLite failed —
        # the file is the source of truth we can recover from later.

    # Also write a JSON archive next to the transcript / eval report.
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ANALYTICS_DIR / f"{transcript_path.stem}-insights.json"
    archive_path.write_text(
        json.dumps(
            {
                "account_number": account_number,
                "customer_id": customer_id,
                "source_transcript": str(transcript_path),
                "insights": payload,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "wrote analytics for account_number=%s to %s", account_number, archive_path
    )
    return archive_path
