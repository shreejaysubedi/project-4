# Provenance Guard — implementation plan

This plan was written before implementation. Scope: the seven required backend
features, a small automated test suite, and documentation. No frontend or optional
stretch features are planned; the submission stays focused on the required API.

## Architecture

```text
Writer
  | POST /submit {text, creator_id}
  v
Flask-Limiter (10/minute and 100/day per source IP)
  | allowed request
  v
Validate JSON, text length, creator ID
  | raw text                    | raw text
  v                             v
Groq semantic assessment     Stylometric measurements
  | AI score + explanation     | AI score + metrics
  +------------------+----------+
                     | two signal objects
                     v
           Weighted score + uncertainty guards
                     | AI-support score, attribution, exact label
                     v
             SQLite transaction
             [content row + immutable decision audit event]
                     | committed decision with content_id
                     v
             JSON response to writer

Creator
  | POST /appeal {content_id, creator_id, creator_reasoning}
  v
Validate input and match creator_id to stored content
  | original decision + reasoning
  v
SQLite transaction: append appeal event; set content.status = under_review
  | original scores + label retained, appeal receipt
  v
JSON response

GET /content/<id> -> current stored decision/status
GET /appeals      -> review queue with original text, signals, label, reasoning
GET /log         -> newest structured audit events (decisions and appeals)
```

A submission passes through rate limiting and validation before both independent
signal functions run; the scorer then produces a label and persists the decision
and audit event together before returning JSON. An appeal links to that stored
decision, changes its current status, and appends an audit event in one transaction
without changing the original attribution or scores.

## Detection signals

1. **Groq semantic assessment (70% weight).** Send text as untrusted user data to
   `openai/gpt-oss-120b`, with a system prompt asking for JSON containing
   `ai_score` (finite number in [0, 1]) and `explanation` (nonempty string).
   This captures context, generic rhetoric, narrative specificity, and stylistic
   coherence holistically. It is useful alongside measurements, but cannot prove
   authorship and can penalize formal or non-native writing. Submitted instructions
   must not be followed. Validate the returned fields; a missing key, API failure,
   timeout, or invalid response produces an unavailable signal, never a made-up
   model result. The course's Llama 4 Scout model is retired; keep the model
   configurable through `GROQ_MODEL`.
2. **Stylometric structure (30% weight).** Pure Python measures sentence-length
   coefficient of variation, vocabulary diversity (type-token ratio), and
   punctuation density. A low sentence-length variation raises the AI score;
   vocabulary diversity and punctuation provide weaker supporting evidence.
   Return `ai_score` plus the metrics and component scores. These are transparent
   structural proxies, not semantic judgments. Repetitive poems, academic prose,
   editing, short passages, and language differences are major blind spots.

The signals measure different properties, but are not statistically independent.
Neither is evidence of who actually created a work. Never remove or penalize a
submission automatically based on the result.

Stylometric components (all clipped to [0, 1]):

- `uniformity = 1 - sentence_length_cv / 0.8`
- `vocabulary = (0.9 - type_token_ratio) / 0.5`
- `punctuation = 1 - punctuation_per_word / 0.3`
- `stylometric_score = 0.75*uniformity + 0.15*vocabulary + 0.10*punctuation`

## Uncertainty representation

`confidence` is an **AI-support index**, not the probability of correct attribution:
0 strongly supports human-like writing, 0.5 is inconclusive, and 1 strongly
supports AI-like writing. A score of 0.6 means weak AI evidence and stays uncertain.
Use `score_kind: "ai_support"` in responses to make the direction explicit.

With both signals available:

```text
raw = 0.70 * llm.ai_score + 0.30 * stylometry.ai_score
agreement = 1 - 0.5 * abs(llm.ai_score - stylometry.ai_score)
confidence = 0.5 + (raw - 0.5) * agreement
```

Apply these guards before assigning the label:

- Fewer than 40 words OR fewer than 3 sentences: clamp confidence to [0.35, 0.65].
- Groq unavailable: `confidence = 0.5 + 0.25*(stylometry.ai_score - 0.5)`,
  necessarily [0.375, 0.625]. Explicitly report degraded analysis.
- `confidence >= 0.85`: `likely_ai`.
- `confidence <= 0.20`: `likely_human`.
- Otherwise: `uncertain`.

Classify the full-precision guarded score, then round the reported score to four
decimals. These are conservative design thresholds, not empirically calibrated
probabilities. A stricter AI threshold reflects the higher harm from falsely
accusing a human creator. A broad middle band, disagreement dampening, and short
text/outage guards keep weak evidence inconclusive.

**Implementation revision after boundary review:** round the guarded score to four
decimals *before* applying thresholds. Otherwise a displayed 0.8500 could carry
an uncertain label because its hidden full-precision value was 0.84999. This small
change keeps the public score and label consistent; the original plan above is
retained here to make the revision explicit.

Validate scoring with known signal pairs and threshold boundaries, then run four
course sample passages (AI example, casual human example, formal human example,
edited AI example). Record each signal separately. Stubbed tests verify software
behavior only; live Groq checks require a real key and must be identified separately.
No accuracy or population-level calibration claim follows from four samples.

## Transparency labels

These exact strings are the API's three label variants and must also appear in
README.md:

- High-confidence AI (`likely_ai`): "Likely AI-generated. Both checks strongly
  suggest AI involvement, but this is not proof. The creator can appeal."
- High-confidence human (`likely_human`): "Likely human-written. Both checks
  suggest human authorship, but this is not verification of who wrote it."
- Uncertain (`uncertain`): "Authorship uncertain. There is not enough consistent
  evidence to distinguish human writing from AI-generated text. No conclusion
  should be drawn."

Scores stay available in the API, while the label uses plain language. An appeal
adds a separate `under_review` status rather than pretending the detector has
changed its conclusion.

## API contract and persistence

- `POST /submit`: JSON `text` (1–20,000 characters after trimming) and `creator_id`
  (1–100 characters). Return 200 with UUID `content_id`, UTC `timestamp`,
  `creator_id`, `attribution`, `confidence`, `score_kind`, `label`, `signals`,
  uncertainty reasons, `analysis_mode`, and `status: "classified"`.
- `POST /appeal`: JSON `content_id`, `creator_id`, and `creator_reasoning`
  (10–4,000 characters). Return 201 with stored decision and appeal details.
- `GET /content/<content_id>`: current decision and appeal status; 404 if missing.
- `GET /appeals`: current review queue; original text is visible to the reviewer.
- `GET /log?limit=20`: JSON `entries`, newest first, limit 1–100.
- `GET /health`: local availability and configuration, without revealing keys.
- Invalid requests return JSON errors (400/413/415), never call Groq, and create no
  decision. Limit requests to 96 KiB. Limiter failures return JSON 429 with
  `Retry-After`; invalid attempts also count against the limit.

SQLite stores content, appeals, and append-only audit events. Every decision
event includes UTC timestamp, content/creator IDs, attribution, confidence, both
signal objects, label, and status. Appeal events copy the original decision's
scores and add appeal ID/reasoning and `status: "under_review"`. Use SQL
parameters and a transaction so an audit write cannot fail silently after a
successful status update. Do not commit the runtime database or secrets.

## Appeals workflow and anticipated edge cases

The original creator submits the content ID, their creator ID, and a concrete
reason. Match the supplied creator ID against the stored one. This is only a
classroom identity check, **not authentication**; a production platform must use
the authenticated creator from its session. The review queue includes the full
original text, unmodified classification, individual signal metrics, timestamp,
and creator reasoning. A human reviewer decides the outcome outside this scope;
automated reclassification and appeal resolution are not required.

1. **Repetitive poem or formal/non-native English:** regular sentences or limited
   vocabulary can raise the structural score. The semantic check, wider uncertain
   band, higher AI threshold, and appeal route reduce harm; they cannot eliminate
   bias. Short poems always get an uncertain result.
2. **Edited AI text or mixed authorship:** editing introduces variation that can
   look human; allow uncertainty and describe AI involvement rather than asserting
   sole authorship. Do not claim this system proves provenance.
3. **Unknown content / mismatched creator / duplicate appeal:** return 404 / 403 /
   409 respectively. A duplicate appeal neither changes reasoning nor appends a
   second audit event; a unique database constraint also handles concurrent calls.
4. **Groq outage or malformed JSON:** keep the structural measurements, clearly
   mark the LLM unavailable, and return only an uncertain classification.
5. **Hostile text containing model instructions:** isolate it in the user message,
   explicitly tell the detector to treat it as data, validate outputs, and document
   that prompt injection cannot be fully ruled out.

## Rate limits

Use Flask-Limiter with `storage_uri="memory://"` and
`10 per minute;100 per day`, keyed by direct source IP. Ten submissions supports
short editing sessions; one hundred daily allows frequent writers while bounding
cost. Do not key solely on user-supplied creator IDs because they are trivial to
rotate. In-memory counters reset on restart and are not shared across workers;
shared IPs can collide. Production needs shared Redis storage, authenticated
creator quotas, and trusted proxy configuration. Add a 10/minute IP limit to
appeals to prevent queue flooding.

## AI Tool Plan

- **M3 — submission + first signal:** Provide Detection signals, API contract,
  persistence, and Architecture. Request a Flask app factory, validation,
  standalone Groq JSON assessment, and transactional audit persistence. Verify
  the signal schema and rejection of invalid model output independently; test
  the route with valid JSON, empty text, and missing/wrong field types.
- **M4 — second signal + scoring:** Provide Detection signals, Uncertainty
  representation, and Architecture. Request pure Python stylometric functions
  and the exact weighting/dampening formula. Check independently computed metrics,
  0.20/0.85 boundaries, disagreement, short text, and missing-model guards. Run
  the four samples and distinguish mocked results from live model output.
- **M5 — production layer:** Provide Transparency labels, Appeals workflow,
  Rate limits, API contract, and Architecture. Request exact labels, appeals,
  review queue, structured logs, and limiter integration. Verify all labels,
  original-decision preservation, persistence after app recreation, ownership
  errors, duplicate appeals, and 12 rapid requests producing ten 200s/two 429s.

## Pre-implementation review

The high-confidence labels explicitly avoid claiming proof, and the uncertain
label does not imply misconduct. Scope remains the required backend; update this
plan before any optional stretch work. README will record actual verification,
limitations, AI assistance, and any implementation differences honestly.
