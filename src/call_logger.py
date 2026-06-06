"""Per-call structured logger.

Attaches event listeners to a live ``AgentSession`` and writes a JSON-Lines
transcript covering everything that happens during the call:

  * caller speech (final STT transcripts)
  * agent speech (assistant chat messages)
  * tool invocations + their outputs
  * close event with reason

Each call produces a single file under ``transcripts/`` named
``<room>-<UTC timestamp>.jsonl``. The path is returned to the caller so the
post-call eval can read it back.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from livekit.agents import AgentSession, JobContext
from livekit.agents.llm.chat_context import ChatMessage

logger = logging.getLogger("call_logger")

# Directory where transcripts land. Configurable via env if ever needed.
TRANSCRIPTS_DIR = Path("transcripts")


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 with a 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _chat_message_text(msg: ChatMessage) -> str:
    """Flatten a ChatMessage's content list into a single string for logging."""
    parts: list[str] = []
    for c in msg.content:
        # Content items are usually plain strings; tool/image parts get repr'd.
        parts.append(c if isinstance(c, str) else repr(c))
    return " ".join(parts).strip()


def register_call_logger(session: AgentSession, ctx: JobContext) -> Path:
    """Wire transcript logging onto the given session.

    Returns the path of the JSONL file. Caller can hand this to the eval
    runner during shutdown.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = _now_iso().replace(":", "-")
    # Sanitize the room name so we don't end up with shell-hostile filenames.
    safe_room = (ctx.room.name or "unknown").replace("/", "_")
    log_path = TRANSCRIPTS_DIR / f"{safe_room}-{started_at}.jsonl"

    # Open in append mode so re-attaching (e.g. on reconnect) doesn't truncate.
    fh = log_path.open("a", encoding="utf-8")

    def write(event: dict[str, Any]) -> None:
        """Serialize a structured event to the JSONL file."""
        event.setdefault("ts", _now_iso())
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    # Header row identifies the call.
    write(
        {
            "event": "session_started",
            "room": ctx.room.name,
            "transcript_path": str(log_path),
        }
    )

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev) -> None:
        """Log the caller's speech. Only final transcripts go to the file."""
        if not ev.is_final:
            return
        write(
            {
                "event": "user_speech",
                "transcript": ev.transcript,
                "speaker_id": ev.speaker_id,
                "language": ev.language,
            }
        )

    @session.on("conversation_item_added")
    def _on_conversation_item(ev) -> None:
        """Log assistant messages that get added to the chat context."""
        item = ev.item
        if isinstance(item, ChatMessage) and item.role == "assistant":
            text = _chat_message_text(item)
            if text:
                write({"event": "agent_speech", "text": text})

    @session.on("function_tools_executed")
    def _on_tools(ev) -> None:
        """Log every tool call with its arguments and output."""
        for call, output in ev.zipped():
            write(
                {
                    "event": "tool_call",
                    "name": call.name,
                    "arguments": call.arguments,
                    "call_id": call.call_id,
                    "output": output.output if output else None,
                    "is_error": bool(output.is_error) if output else None,
                }
            )

    @session.on("close")
    def _on_close(ev) -> None:
        """Log the close event and shut the file handle cleanly."""
        write(
            {
                "event": "session_closed",
                "reason": str(ev.reason),
                "error": str(ev.error) if ev.error else None,
            }
        )
        try:
            fh.close()
        except Exception:
            logger.exception("failed to close transcript file %s", log_path)

    logger.info("call logger attached; writing transcript to %s", log_path)
    return log_path
