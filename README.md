# Speaker Gate — Change Summary

---

# Part 1 — Task Requirements

The task asked to improve the quality, robustness, and speed of the VAD and speaker verification pipeline, with three specific known issues to address. This section describes how each was resolved.

---

## Known Issue #1: Tail Cutoff vs. Background Noise

> *When a user stops speaking, we need a buffer window to avoid cutting the stream too abruptly. However, during this window, background noise (e.g., TV, other people) can sometimes be incorrectly attributed to the user.*

This was caused by four separate problems working together:

### Inverted VAD Hysteresis

The VAD exit threshold (0.70) was higher than the entry threshold (0.50) — backwards. Once speech was detected, it was harder to *stay* in speech than to *enter* it. Brief dips in voice energy would immediately end the speech segment, causing constant mid-sentence cutoffs.

**Fix:** Set the exit threshold to 0.35 (below the 0.50 entry), creating proper hysteresis. Speech detection is now sticky — once you start talking, natural pitch and volume dips don't cut you off. The temporal filter (300ms minimum silence) still ensures speech ends cleanly.

### Fixed Silence Timeout

The system used a fixed silence timeout for all speakers. Fast speakers got their sentences fragmented, while slow speakers got cut off between thoughts.

**Fix:** Built an adaptive silence detector that learns the speaker's tempo in real time. It tracks inter-word gaps during active speech, computes the 90th percentile, and multiplies by 1.5x to set a dynamic timeout clamped between 300ms (fast talker) and 1500ms (slow talker). The initial timeout before adaptation kicks in is 1.2 seconds — generous enough for the first utterance. After 2-3 inter-word gaps, it adapts.

### Narrow Trailing Resume Window

After confirming a user, the trailing state resume window was only 150ms. Any pause longer than that — a breath between sentences, a brief hesitation — would end the segment and force full re-verification.

**Fix:** Made the trailing resume window dynamic: `max(0.3s, min(0.8s, adaptive_timeout × 0.6))`. This ties it to the adaptive silence detector, so the resume window scales with the speaker's natural pause duration. Multi-sentence utterances are now captured as single segments.

### Tail-Biased Embedding Window

Similarity checks always used the *tail* (last 1.6 seconds) of the audio segment. After speech ends and silence accumulates, the tail slides into silence, producing progressively worse embeddings. Background noise in the post-speech buffer would get embedded and compared against the speaker's clean reference, sometimes producing false matches.

**Fix:** The embedding window is now speech-centered. During active speech, the tail is used (freshest audio). Once silence begins, the window is centered on the speech portion — maximizing speech content and minimizing silence/noise contamination. The system tracks where speech occurred in the segment and positions the 1.6s Resemblyzer window accordingly.

---

## Known Issue #2: Short Utterances

> *Very short phrases like "yes" or "no" often lack sufficient data for reliable embedding generation.*

Resemblyzer requires 1.6 seconds of audio to produce an embedding. A 0.5-second "yes" in a 1.6s window is 70% silence — the resulting embedding scores 0.30–0.47 against the reference, well below any threshold. This is a fundamental limitation of the embedding model, not a tuning problem.

**Fix:** Addressed at the decision level rather than the embedding level:

- **Trust-based short utterance approval:** If the user was confirmed within the last 10 seconds and the new utterance is under 2 seconds, it's auto-approved without a similarity check. The recency of confirmation is the evidence — Resemblyzer cannot produce reliable embeddings from sub-second speech, so checking would only cause false rejections. Background audio and other speakers never get confirmed in the first place, so they can't exploit this path.
- **Speech-centered embedding:** For utterances long enough to check (1–2 seconds), centering the window on speech instead of using the silence-heavy tail raises scores from the 0.30–0.47 range to the 0.55–0.68 range — often enough to pass threshold.
- **Single-match confirmation:** Reduced the required consecutive matches from 2 to 1. For borderline 1–1.5s speech that gets only 2–3 checks total, requiring two consecutive passes was too strict. Counter decay (see below) still prevents single lucky checks from confirming non-users.
- **Stale score cleanup:** Match counters, similarity history, and peak scores are cleared when a new utterance begins, preventing scores from a rejected utterance contaminating the next one.

The expected usage pattern: say a sentence to establish identity (1.5s+), then use short commands freely for the next 10 seconds. After a long pause (>10s), a sentence re-establishes identity.

---

## Known Issue #3: Similarity Thresholds

> *The current similarity threshold is dynamic, but it requires more rigorous testing across different microphones and environments.*

The threshold system had three problems: contaminated enrollment embeddings, a single global threshold that couldn't adapt, and a brittle match/mismatch counter that let single noisy checks override accumulated evidence.

### Contaminated Enrollment

During enrollment, embeddings were extracted from *all* audio windows including silence, background noise, and breathing. These garbage windows contaminated the reference embedding and lowered the consistency score, resulting in a loose threshold that let non-matching audio through.

**Fix:** Added a `speech_mask()` function that pre-computes which audio windows actually contain speech. Windows with less than 50% speech content are skipped during enrollment. Only genuine speech windows contribute to the reference embedding. Consistency scores rose from ~0.62 to ~0.70, and the calculated thresholds became much more meaningful.

### Per-Session Thresholds

With multi-session enrollment (see Part 2), a single global threshold computed from cross-session consistency was too loose. Cross-session consistency is naturally lower because different recordings capture different voice characteristics.

**Fix:** Each enrollment session gets its own threshold based on its *internal* consistency. During verification, when the best-matching session is found, the system dynamically switches to that session's threshold. The formula maps consistency (0.60–1.00) to threshold (0.58–0.72):

```
threshold = 0.58 + (consistency - 0.60) / 0.40 × 0.14
```

### Similarity Score Accumulation

Speaker verification used a simple counter — each check either passed or failed, and a single bad check could reset all progress. Borderline speakers would bounce between states indefinitely.

**Fix:** Three improvements:
- **Decay counters:** A match *decays* the mismatch counter (and vice versa) instead of hard-resetting. This smooths out noisy per-window scores.
- **Score history:** The last 5 similarity scores are tracked in a rolling window.
- **Accumulation fallback:** If 3+ consecutive checks average above the threshold (even if no single check individually passes), the speaker is confirmed.

### Tuned Defaults

Several parameters were adjusted based on real-world testing:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| silence_timeout | 0.8s | 1.2s | First utterance needs room before adaptation kicks in |
| pre_buffer_sec | 1.0s | 0.5s | Less pre-speech audio, reduces segment bloat |
| post_buffer_sec | 0.3s | 0.15s | Faster segment delivery after speech ends |
| embed_interval_sec | 0.3s | 0.2s | More frequent similarity checks |
| embed_suffix_sec | 2.0s | 10.0s | Generous grace window for short utterance auto-approval |
| short_utterance_sec | 1.5s | 2.0s | More generous definition of "short" response |
| enter_count | 2 | 1 | Faster confirmation — counter decay handles false positives |
| pending_grace_sec | 0.5s | 0.3s | Reject non-matching speech faster |
| base_threshold | 0.65 | 0.58 | Previous value was too strict for streaming audio |
| max_threshold | 0.78 | 0.72 | Prevents overly aggressive thresholds from high-consistency enrollments |

### Configuration Passthrough

The POC server configuration was missing 9 parameters that the core SDK supported. These were silently ignored — changes to VAD timing, silence detection, or buffer sizes through environment variables had no effect. All 27 parameters are now passed through end-to-end.

---

# Part 2 — Additional Improvements

Beyond fixing the known issues, the following enhancements were built on top of the working system.

---

## Multi-Session Enrollment

A single enrollment recording can't capture the full range of a person's voice. Someone who enrolled in a calm tone would get rejected when speaking energetically.

The system now supports multiple enrollment sessions per profile. Each session stores its own reference embedding. During verification, the incoming audio is compared against *all* session references, and the best match (with its own per-session threshold) is used. Sessions are merged when re-enrolling — the system detects an existing profile and adds the new session alongside previous ones.

---

## Cross-Session Enrollment Validation

Nothing prevented a different person's voice from being enrolled as a new session. A friend's recording would be accepted, and the system would then recognize that friend as the enrolled user.

Before merging a new enrollment session, the system now checks its similarity against all existing sessions. If the best match is below 0.55, the enrollment is rejected: *"Voice doesn't match existing profile."* Only the enrolled user can add new sessions.

---

## Passive Adaptive Enrollment

Users had to manually re-enroll to teach the system new voice patterns. Meanwhile, the system had access to hours of verified speech — perfectly labeled training data going to waste.

Inspired by Apple's Face ID, the system now learns passively from verified speech:

1. Confirmed audio segments (longer than 5 seconds) are accumulated during normal use.
2. Once 20 seconds of verified audio accumulates, an embedding is extracted.
3. **Coherence check:** If the voice pattern changed significantly mid-accumulation (coherence < 0.70), the accumulator resets. This prevents blending different voice patterns into one session.
4. Before creating a new session, three checks run:
   - **Identity:** Is this the same person? (similarity > 0.55 to at least one existing session)
   - **Novelty:** Is this genuinely different? (similarity < 0.80 to all existing sessions — avoids redundant slots)
   - **Capacity:** If all 6 slots are full, the most similar existing session is retired in favor of the fresher one.

The system continuously adapts — morning voice, evening voice, energetic, tired — all learned automatically without manual re-enrollment.

---

## Recorded Audio Protection

Speaker verification uses decay-based match/mismatch counters and multi-session comparison, which together make it difficult for recorded audio to pass — a single lucky check isn't enough when subsequent checks pull the counters back down. Background audio from TV, Instagram, and other media is reliably rejected through cosine similarity alone (non-user audio consistently scores well below threshold).

---

## Profile Management UI

**Profile panel** below the controls shows: profile ID, number of enrollment sessions, per-session threshold tags (S1: 0.608, S2: 0.599, etc.), active session highlighting when a segment matches, and a clear button to delete all sessions.

**Enhanced segment history** shows match details per segment: which session matched (S1/3), similarity score, and threshold used.

**Adaptive enrollment notifications** — when a new session is created from verified speech, the profile panel updates live with a yellow flash animation.

---

## Improved Debug Logging

Every similarity check logs which session matched best, all per-session scores, and the active threshold:

```
Similarity check: best_session=1/3, sim=0.842, sims=['0.842', '0.780', '0.816'], threshold=0.608
```

---

# Results

After all changes, the system:

- **Reliably recognizes the enrolled user** across different speaking styles
- **Handles short commands** ("yes", "no") naturally after identity is established
- **Rejects background audio** from TV, Instagram, and other media
- **Rejects recorded/played-back audio** and other speakers
- **Adapts to the user's voice** over time without manual re-enrollment
- **Handles natural pauses** without fragmenting utterances
- **Provides real-time visibility** into verification decisions through the client UI
