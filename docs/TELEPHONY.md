# Telephony — Call the Agent from Your Phone

This guide walks you through wiring a real phone number to the ABC Bank
voice agent so you can dial in and talk to it like any other support
line. We use the most common free path: a **Twilio trial account**
($15 free credit, enough for a US number + ~50 minutes of test calls)
fronted by **LiveKit Cloud's SIP service**.

```
   📞 your phone  ──► Twilio number ──► Twilio Elastic SIP Trunk
                                                │
                                                ▼
                                  LiveKit Cloud SIP inbound URI
                                                │
                                                ▼
                                  LiveKit Cloud Inbound Trunk
                                                │
                                                ▼
                                  Dispatch rule → room created
                                                │
                                                ▼
                                  Your `my-agent` worker joins
                                                │
                                                ▼
                                      Phase 1 greeting plays
```

You don't need to change a single line of the agent's logic for this —
LiveKit Cloud's agent dispatcher automatically joins your worker to any
room a SIP call creates. The one tweak we made is an `on_enter` hook so
the agent greets the caller as soon as the line opens (rather than
waiting for the caller to say "hello" first).

---

## Prerequisites

1. **LiveKit CLI** version 2.15.0 or later. Install:
   ```bash
   # macOS
   brew install livekit-cli
   # Linux
   curl -sSL https://get.livekit.io/cli | bash
   # Windows
   winget install LiveKit.LiveKitCLI
   ```
   Then authenticate against your LiveKit Cloud project:
   ```bash
   lk cloud auth
   ```

2. **The agent deployed to LiveKit Cloud** (so a worker is online to
   answer incoming dispatches). From the repo root:
   ```bash
   # Loads LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET into .env.local
   lk app env -w -d .env.local
   # Build and deploy
   lk agent deploy
   ```
   Confirm the agent worker is running:
   ```bash
   lk agent status
   ```

3. **A Twilio trial account** (free): https://www.twilio.com/try-twilio
   The trial gives you ~$15 of credit and one free trial phone number.

---

## Step 1 — Get a free Twilio phone number

1. Sign in to https://console.twilio.com.
2. Twilio will offer to give you a free trial number during onboarding —
   accept it. (If you skipped it: **Phone Numbers → Manage → Buy a
   number**, filter for ones that cost $0.00 on the trial.)
3. Take note of the E.164 number, e.g. `+15551234567`.

> **Trial-account caveat**: Twilio trial numbers can only call/receive
> from "verified caller IDs". Verify your own mobile number under
> **Phone Numbers → Verified Caller IDs** before you try to dial in.

---

## Step 2 — Get LiveKit's SIP termination URI

LiveKit Cloud assigns each project a unique inbound SIP host. Look it up:

```bash
lk sip inbound-uri
```

You'll get something like:

```
sip:<your-project>.sip.livekit.cloud
```

Keep this handy — you'll paste it into Twilio in the next step.

---

## Step 3 — Create a Twilio Elastic SIP Trunk pointing at LiveKit

In the Twilio console:

1. **Elastic SIP Trunking → Manage → Trunks → Create new SIP Trunk**.
2. Friendly name: `livekit-abcbank`.
3. Open the new trunk, click **Origination**, then **Add new Origination URI**.
   * URI: `sip:<your-project>.sip.livekit.cloud` (from Step 2).
   * Priority/weight: leave as 10 / 10.
   * Save.
4. Click **Numbers** on the same trunk, **Add an Existing Number**, and
   pick your trial number.

That's the Twilio side done. Inbound calls to your number will now route
to LiveKit Cloud as SIP traffic.

---

## Step 4 — Tell LiveKit to accept that number (Inbound Trunk)

LiveKit needs to know which phone numbers belong to you so it knows
which SIP traffic is yours. There's a starter file at
`sip/inbound-trunk.json` — replace the placeholder with your Twilio
number (still in E.164, e.g. `+15551234567`):

```bash
$EDITOR sip/inbound-trunk.json
lk sip inbound create sip/inbound-trunk.json
```

Note the **trunk ID** it prints (looks like `ST_…`). You'll use it
in the next step.

---

## Step 5 — Tell LiveKit how to dispatch incoming calls to the agent

The dispatch rule says "when a call hits this trunk, create a room and
ask the `my-agent` agent worker to join it". The starter file is at
`sip/dispatch-rule.json`. Drop your trunk ID into it:

```bash
$EDITOR sip/dispatch-rule.json   # paste your ST_… into trunk_ids
lk sip dispatch create sip/dispatch-rule.json
```

The agent name in the JSON (`my-agent`) matches the
`@server.rtc_session(agent_name="my-agent")` decorator in
`src/agent.py`. If you rename one, rename the other.

---

## Step 6 — Dial your number

Call the Twilio number from your phone (the same mobile you verified
in Step 1, if you're on a trial account). Within a couple of seconds
you should hear:

> "Thank you for calling ABC Bank, what can I help you with today?"

That's the agent's `on_enter` hook firing as soon as the SIP call
connects.

While you're on the call, in another terminal you can watch the agent
worker logs and confirm STT/LLM/tool events:

```bash
lk agent logs --tail
```

After you hang up, the post-call hooks fire on the worker:

* `transcripts/abcbank-call-<id>.jsonl` — the structured transcript
* `evals/…-eval-….md` — Gemini scoring of phase + guardrail adherence
* `analytics/…-insights.json` + `customer_insights` row in SQLite —
  marketing-ready insights keyed by the provisioned `account_number`

---

## Step 7 — Verify and iterate

Useful commands while testing:

```bash
# Worker status / logs
lk agent status
lk agent logs --tail

# SIP plumbing
lk sip inbound list
lk sip dispatch list
lk sip participant list   # currently-active SIP calls

# Inspect a finished call's outputs locally (after `lk agent logs` shows the call ended)
ls -lt transcripts evals analytics
```

If the call connects but the agent stays silent:
* Confirm `lk agent status` shows the worker as running.
* Confirm the dispatch rule's `agent_name` matches `"my-agent"`.
* Tail `lk agent logs` while you redial — you should see "agent joined
  room" and then `agent_speech` events.

If the call doesn't even connect:
* Twilio console → your trunk → **Recent Calls**: shows the SIP error.
* Most common: trial-account number not verified, or the Origination URI
  was typed wrong.

---

## Costs (free? mostly)

| Item                                    | Cost on trial |
|-----------------------------------------|---------------|
| Twilio trial credit                     | $15 free      |
| Trial phone number                      | $0            |
| Twilio inbound voice (US)               | ~$0.0085/min  |
| LiveKit Cloud free tier                 | $0            |
| LiveKit Inference (LLM/STT/TTS)         | metered, but [free tier covers heavy testing](https://livekit.io/pricing) |
| Gemini API for eval/analytics           | [free tier](https://ai.google.dev/pricing) covers many calls/day |

The $15 Twilio credit is enough for ~1,750 minutes of inbound voice on
US numbers, which is more than you'll burn through testing.

---

## Alternative free providers

If Twilio's trial restrictions are inconvenient, the same setup works
with any SIP-capable carrier. Two free-credit alternatives:

* **Telnyx** — sign up at https://telnyx.com (account credit varies).
  Configure an Outbound Voice Profile + SIP connection pointing to the
  same LiveKit termination URI.
* **Vonage** — https://www.vonage.com/communications-apis (trial
  credit). Same flow.

LiveKit's docs cover both in detail:
https://docs.livekit.io/sip/quickstart/.

---

## What's deployed where

Code-side, none of this changed except the `on_enter` greeting hook.
The complete telephony surface lives in three places:

| File / artifact                       | What it does                                |
|---------------------------------------|---------------------------------------------|
| `src/agent.py` (`Assistant.on_enter`) | Speaks the greeting when the call connects  |
| `sip/inbound-trunk.json`              | Tells LiveKit which numbers are yours       |
| `sip/dispatch-rule.json`              | Routes incoming calls to the `my-agent` worker |
| Twilio Elastic SIP Trunk              | Forwards your phone number to LiveKit       |

Nothing about the workflow, guardrails, tools, transcript logging,
eval, or analytics changes — they all just see a normal LiveKit room.
