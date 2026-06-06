# ABC Bank Voice Agent — System Architecture

End-to-end design for the autonomous voice agent that opens new checking
accounts for ABC Bank over the phone.

---

## 0. Tech jargon for newcomers

If terms like "STT" or "function tool" are new, read this first. Everything
below uses these words constantly.

| Term | What it means in plain English |
|---|---|
| **Voice agent** | A program that listens to a caller, decides what to say, and talks back — like a phone-based ChatGPT with a job to do. |
| **LLM** (Large Language Model) | The "brain" that reads what the caller said and writes the reply. Here we use OpenAI's GPT family via LiveKit Inference. |
| **STT** (Speech-to-Text) | Microphone audio → written words. We use Deepgram Nova-3. |
| **TTS** (Text-to-Speech) | Written words → spoken audio. We use Cartesia Sonic-3. |
| **VAD** (Voice Activity Detection) | A tiny model that decides "is the caller talking right now or just breathing?" — keeps the agent from interrupting. We use Silero VAD. |
| **Turn detector** | Decides "has the caller finished their sentence?" so the agent knows when it's its turn to talk. We use LiveKit's Multilingual model. |
| **System prompt** | A big text block of permanent instructions given to the LLM at the start of every call. The agent's "personality and rulebook". |
| **Function tool / tool call** | A Python function the LLM is allowed to invoke (e.g. "save this customer to the database"). The LLM cannot touch your code or DB directly — it only requests tools by name. |
| **Tool schema** | A machine-readable description of a function's name, arguments, and what it does. The LLM reads this to know what tools exist. |
| **WebRTC / SIP** | The plumbing that carries the actual audio over the internet (WebRTC for web/mobile apps, SIP for normal phone numbers). |
| **LiveKit Cloud** | Managed servers that handle WebRTC, SIP, and the agent runtime so we don't run our own infrastructure. |
| **LiveKit Inference** | LiveKit's gateway to AI providers. One credential instead of separate keys for OpenAI/Cartesia/Deepgram. |
| **Agent worker** | The long-running Python process (`src/agent.py`) that joins a call and orchestrates STT → LLM → TTS. |
| **Room / session** | A single call. Each caller gets a fresh "room" with its own agent instance. |
| **Context window** | Everything the LLM "remembers" within one call: system prompt + transcript so far + tool results. There's no memory between calls. |
| **Guardrails** | Hard rules embedded in the system prompt that the agent must never violate (e.g. don't echo PII, don't give financial advice). |
| **PII** (Personally Identifiable Information) | Data that identifies a person — name, address, DOB, ID number, etc. Subject to privacy rules. |
| **Prompt injection** | When a caller tries to trick the agent with phrases like "ignore all previous instructions". Defenses live in the guardrails. |
| **Function calling loop** | LLM picks a tool → runtime runs it → result goes back to the LLM → LLM speaks or picks the next tool. Repeats until the call ends. |

---

## 1. What the agent does

The agent is a single-purpose phone receptionist for ABC Bank. It can do
exactly **one** job: open a brand-new checking account for a caller.
Everything else — balance inquiries, transfers, loans, existing-account
help — it politely declines and tells the caller to visit a branch.

Within the account-opening job, the agent runs an end-to-end workflow:

1. Answers the call and asks what the caller wants.
2. Pitches the offer (no monthly fees for the first year, ATM access) and
   reads the standard terms.
3. Obtains explicit verbal consent.
4. Collects 10 required profile fields with per-field confirmation.
5. Saves the verified profile to a SQLite database.
6. Provisions a real checking account (account number + routing number) via
   a Core Banking API call.
7. Optionally emails the caller a welcome packet with their Account ID,
   login URL, and a temporary password.
8. Confirms success verbally and ends the call.

Throughout the call it enforces strict guardrails: no financial advice, no
PII echoed back over the audio channel, no tools triggered before consent
is given and every field is verified.

---

## 2. Capabilities

| # | Capability | How it's delivered |
|---|---|---|
| 1 | Inbound voice conversation | LiveKit Cloud + LiveKit Agents Python SDK |
| 2 | Real-time speech recognition | Deepgram Nova-3 STT (multilingual) |
| 3 | Natural spoken responses | Cartesia Sonic-3 TTS, female voice |
| 4 | Knows when to listen vs. speak | Silero VAD + LiveKit Multilingual turn detector |
| 5 | Background-noise suppression | LiveKit Cloud noise cancellation (ai-coustics QUAIL VF-S) |
| 6 | Workflow execution (5–7 phases) | System prompt driving an LLM (`openai/gpt-5.2-chat-latest`) |
| 7 | Customer data persistence | `collect_customer_information` tool → SQLite (`pending_customers` table) |
| 8 | Bank account provisioning | `provision_bank_account` tool → mock Core Banking API |
| 9 | Welcome email with credentials | `send_welcome_email` tool → SMTP (or log fallback) |
| 10 | Guardrails: scope, PII masking, consent gate, anti-jailbreak | Embedded directives in the system prompt |
| 11 | Confirmation loop on every field | Prompt instruction in Phase 3 |
| 12 | Preemptive generation (low latency) | `AgentSession(preemptive_generation=True)` |
| 13 | Production deploy target | Dockerfile → LiveKit Cloud |

---

## 3. Process (call lifecycle)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CALL LIFECYCLE                           │
└─────────────────────────────────────────────────────────────────┘

  Caller dials in (PSTN/SIP or web client)
            │
            ▼
  ┌──────────────────────┐
  │  LiveKit Cloud Room  │   (WebRTC media transport)
  └──────────┬───────────┘
             │  audio frames
             ▼
  ┌──────────────────────┐
  │   Agent Worker       │   src/agent.py, `start` command
  │   (Python process)   │
  └──────────┬───────────┘
             │
   ┌─────────┴──────────────────────────────────────┐
   │           AgentSession orchestrator             │
   │                                                 │
   │   STT  ──▶  Turn Detector  ──▶  LLM  ──▶  TTS  │
   │   (Deepgram)   (Multilingual)  (GPT)  (Cartesia)│
   │       ▲                          │              │
   │       │                          ▼              │
   │   Silero VAD               Tool calls           │
   │                                                 │
   └─────────────────────────────────────────────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │     Tool registry      │
                       │  (banking_tools)       │
                       └─────────┬──────────────┘
                                 │
            ┌────────────────────┼─────────────────────┐
            ▼                    ▼                     ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ collect_customer │ │ provision_bank_  │ │ send_welcome_    │
   │ _information     │ │ account          │ │ email            │
   ├──────────────────┤ ├──────────────────┤ ├──────────────────┤
   │ database.py      │ │ core_banking.py  │ │ email_dispatch.py│
   │  → SQLite        │ │  → mock API +    │ │  → SMTP or log   │
   │   pending_       │ │    DB update     │ │                  │
   │   customers      │ │                  │ │                  │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
```

### Phase-by-phase flow

```
[Phase 1] ────────────────────────────────────────────────────────
  Receptionist greeting & intent routing (system prompt only).
  Agent speaks: "Thank you for calling ABC Bank, what can I help
                you with today?"
  ├─ Account opening? → Phase 2
  └─ Anything else?  → Out-of-scope script → hang up

[Phase 2] ────────────────────────────────────────────────────────
  Offer details + consent gate (system prompt only).
  Agent reads offer + asks for verbal "yes".
  ├─ Yes → Phase 3
  └─ No  → Graceful termination script → hang up

[Phase 3] ────────────────────────────────────────────────────────
  Iterative data collection (system prompt only).
  Collects 10 fields one at a time:
    1. First & last name        6. Phone number   (PII)
    2. Age                      7. Email          (PII)
    3. Residential address(PII) 8. Citizenship status
    4. ID number          (PII) 9. Employment status
    5. Date of birth      (PII) 10. Account opening goal (confirm)
  ── Confirmation loop on every field. PII fields use masked
     acknowledgment ("Thank you, I have recorded that.") and a
     non-echoing confirm. Non-PII fields are repeated back.
  ── Validation gate: cannot leave Phase 3 until all 10 confirmed.

[Phase 4] ────────────────────────────────────────────────────────
  TOOL CALL: collect_customer_information(...)
  → database.save_pending_customer() inserts row, returns customer_id

[Phase 5] ────────────────────────────────────────────────────────
  TOOL CALL: provision_bank_account(customer_id)
  → core_banking.create_bank_account() generates account_number,
    attaches account_number + routing_number to the DB row,
    returns both to the LLM.

[Phase 6] ────────────────────────────────────────────────────────
  Opt-in: agent asks "Would you like an email confirmation?"
  ├─ Yes → TOOL CALL: send_welcome_email(to, name, account_number,
  │                                       routing_number)
  │        → email_dispatch generates a 12-char temp password,
  │          composes body w/ Account ID + login URL + temp password,
  │          sends via SMTP (or logs in dev).
  └─ No  → Skip email; mention branch alternative.

[Phase 7] ────────────────────────────────────────────────────────
  Final confirmation script (system prompt only).
  Agent speaks closing line → call terminates.
```

---

## 4. Component architecture

```
┌──────────────────────────── CLIENT EDGE ───────────────────────────────┐
│                                                                        │
│   📞 PSTN caller                  💻 Web/mobile frontend                │
│        │                                  │                            │
│        └──────── SIP trunk ───────┐       └─── WebRTC ───┐             │
│                                   ▼                      ▼             │
└──────────────────────────────────────────────────────────────────────  │
                                                                        │
┌──────────────────────────── LIVEKIT CLOUD ────────────────────────────┐│
│                                                                       ││
│  • Media server (WebRTC SFU)                                          ││
│  • Background noise cancellation                                      ││
│  • LiveKit Inference gateway (LLM/STT/TTS routing)                    ││
│  • Agent dispatch                                                     ││
│                                                                       ││
└──────────────────────────────────────────────────────────────────────┘│
                                                                        │
┌──────────────────────────── AGENT WORKER ─────────────────────────────┐│
│  (Dockerized Python process — src/agent.py)                           ││
│                                                                       ││
│  ┌──────────────────── AgentSession ──────────────────────────────┐   ││
│  │                                                                │   ││
│  │   STT pipeline ◄─ VAD ◄─ inbound audio                         │   ││
│  │       │                                                        │   ││
│  │       ▼                                                        │   ││
│  │   Turn detector                                                │   ││
│  │       │                                                        │   ││
│  │       ▼                                                        │   ││
│  │   LLM (GPT) ◄────── system prompt + transcript + tool results  │   ││
│  │       │                                                        │   ││
│  │       ├──► function-calling loop ──► tool registry             │   ││
│  │       │                                                        │   ││
│  │       ▼                                                        │   ││
│  │   TTS pipeline ─► outbound audio                               │   ││
│  └────────────────────────────────────────────────────────────────┘   ││
│                                                                       ││
│  ┌─────────────── tool registry (banking_tools) ─────────────────┐    ││
│  │   collect_customer_information                                │    ││
│  │   provision_bank_account                                      │    ││
│  │   send_welcome_email                                          │    ││
│  └───────────────────────────────────────────────────────────────┘    ││
│                                                                       ││
└───────────────────────────────────────────────────────────────────────┘│
        │                       │                         │              │
        ▼                       ▼                         ▼              │
┌────────────────┐    ┌────────────────────┐   ┌────────────────────┐   │
│ SQLite         │    │ Core Banking API   │   │ SMTP server        │   │
│ pending_       │    │ (mocked locally)   │   │ (or local logger)  │   │
│ customers      │    │                    │   │                    │   │
└────────────────┘    └────────────────────┘   └────────────────────┘   │
                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

### Module map

| File | Responsibility |
|---|---|
| `src/agent.py` | Agent definition, system prompt, tool definitions, AgentSession wiring, entrypoint. |
| `src/database.py` | SQLite schema + `save_pending_customer` + `attach_account_numbers`. |
| `src/core_banking.py` | Mock Core Banking API; generates account numbers and writes them back to the DB. |
| `src/email_dispatch.py` | Welcome email template, temp-password generator, SMTP send (with log fallback). |
| `tests/test_agent.py` | Behavior evals using `AgentSession` test harness. |
| `pyproject.toml` | Pinned to `livekit-agents[silero,turn-detector]==1.5.16`, `livekit-plugins-ai-coustics`. |
| `Dockerfile` | Production image targeting LiveKit Cloud. |

---

## 5. System prompt structure (`Assistant.instructions`)

The prompt is the single source of truth for what the agent does. It is
organized into stacked layers, each taking precedence over those above:

```
┌─────────────────────────────────────────────────────────┐
│  Persona & tone (warmth, patience, no frustration)      │
├─────────────────────────────────────────────────────────┤
│  Output rules (voice-friendly formatting)               │
├─────────────────────────────────────────────────────────┤
│  Speech formatting (read identifiers with pauses)       │
├─────────────────────────────────────────────────────────┤
│  Workflow phases 1–7 (scripts + tool gating)            │
├─────────────────────────────────────────────────────────┤
│  Tool usage notes                                       │
├─────────────────────────────────────────────────────────┤
│  GUARDRAILS (override everything above and any user)    │
│   1. Scope & identity (no financial advice, OOS)        │
│   2. Data privacy (PII masking, consent enforcement)    │
│   3. Tool execution boundaries (no hallucinated args)   │
│   4. Anti-jailbreak (prompt injection, 3-turn timeout)  │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Data flow & storage

### `pending_customers` schema (SQLite)

```
id                      INTEGER PRIMARY KEY AUTOINCREMENT
first_name              TEXT
last_name               TEXT
age                     INTEGER
residential_address     TEXT
identification_number   TEXT
date_of_birth           TEXT
phone_number            TEXT
email                   TEXT
citizenship_status      TEXT
employment_status       TEXT
confirmed_goal          TEXT
account_number          TEXT   ── populated in Phase 5
routing_number          TEXT   ── populated in Phase 5
provisioned_at          TIMESTAMP
created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

The database file path is configurable via `BANKING_DB_PATH`
(default: `banking.db`). It is git-ignored.

### What leaves the agent worker

| Destination | What goes there | When |
|---|---|---|
| LiveKit Cloud (audio) | TTS frames | Continuously during the call |
| SQLite (local) | Verified profile | End of Phase 4 |
| SQLite (local) | account_number, routing_number | End of Phase 5 |
| SMTP (or log) | Welcome email body | End of Phase 6, if opted-in |

PII never leaves the worker over the audio channel by design — see PII
Masking guardrail.

---

## 7. Tech stack summary

| Layer | Choice | Why |
|---|---|---|
| Voice transport | LiveKit Cloud (WebRTC, SIP) | Managed media + noise cancellation + agent dispatch |
| Agent framework | `livekit-agents==1.5.16` | Built-in `AgentSession`, function tools, evals |
| LLM | `openai/gpt-5.2-chat-latest` via LiveKit Inference | Strong instruction-following; no extra API keys |
| STT | Deepgram Nova-3 (multilingual) | High accuracy on names & numbers |
| TTS | Cartesia Sonic-3 | Low latency, warm-sounding voice |
| Turn detection | LiveKit Multilingual model | Avoids cutting the caller off |
| VAD | Silero | Standard, lightweight |
| Noise cancellation | `ai_coustics` QUAIL VF-S | Improves STT in real-world calls |
| Persistence | SQLite (stdlib `sqlite3`) | Zero-config for dev; swap for Postgres in prod |
| Email | stdlib `smtplib` + `EmailMessage` | Simple, swappable with SES/SendGrid later |
| Packaging | `uv` + `pyproject.toml` | Reproducible installs |
| Deployment | Docker → LiveKit Cloud | One-step `lk agent deploy` |
| Telephony | Vapi or Twilio SIP → LiveKit | Inbound phone number → SIP trunk → room |

---

## 8. Failure modes & how the system handles them

| Failure | Detection | Recovery |
|---|---|---|
| Inaudible / ambiguous answer | LLM judgment + guardrail "no hallucination" rule | Ask caller to repeat or spell |
| Caller refuses consent in Phase 2 | LLM follows consent gate | Speaks graceful termination script, ends call |
| DB insert raises | `collect_customer_information` returns `ERROR:` payload | LLM apologizes briefly, retries tool call |
| Account provisioning raises | `provision_bank_account` returns `ERROR:` payload | LLM apologizes, retries |
| Email send raises | `send_welcome_email` returns `ERROR:` payload | LLM tells caller the email failed, offers to retry |
| Caller off-topic / silent | 3-turn timeout guardrail | Speaks "I am unable to assist you further at this time. Goodbye." and ends call |
| Prompt injection attempt | Guardrail directive | Speaks canned reply: "I am here to assist with opening a bank account…" |

---

## 9. Future architecture: multi-agent

Planned next step (not yet implemented). The single Assistant becomes a
Receptionist router that hands off to specialized sub-agents based on
intent:

```
              ┌──────────────────────┐
              │  Receptionist Agent  │
              │  (greeting + intent) │
              └──────────┬───────────┘
                         │
   ┌─────────────────────┼─────────────────────┐
   ▼                     ▼                     ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Account        │  │ Fraud /        │  │ General FAQ /  │
│ Creation Agent │  │ Suspicion      │  │ Branch routing │
│ (today's flow) │  │ Agent          │  │                │
└────────────────┘  └────────────────┘  └────────────────┘
```

LiveKit Agents supports this via the handoff API on `AgentSession`. Each
sub-agent has its own narrower system prompt + tool registry, which
reduces context bloat and latency.
