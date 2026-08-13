# ITEM Batch Protocol Investigation Plan

> **For agentic workers:** Execute inline in the current Phase 2 workspace. Do not dispatch subagents. Follow `superpowers:systematic-debugging`, `superpowers:test-driven-development`, and `superpowers:verification-before-completion`.

**Goal:** Explain and reduce real multi-ITEM batch protocol failures without weakening the unambiguous mapping between ITEM IDs and subtitle events.

**Architecture:** Preserve the current strict batch parser and treat Ollama as an external boundary. First classify every real malformed response and reproduce dominant patterns under controlled batch size, position, neighbors, envelope format, and identifier format. Only after one causal mechanism is confirmed, add a regression fixture from a real response and implement the smallest recovery strategy that still requires valid IDs and complete envelopes.

**Tech Stack:** Python 3, asyncio, pytest, Ollama `qwen2.5:14b`, JSON/JSONL diagnostic artifacts, ASS/SRT format handlers.

## Global Constraints

- Do not modify strict individual bare-body recovery unless a direct regression test requires it.
- Do not modify segmented `\N` recovery unless a direct regression test requires it.
- Batch responses continue to require ITEM envelopes.
- Never map translations by order, line number, or response count when IDs are absent or ambiguous.
- Do not change the multilingual global prompt without causal evidence.
- Keep cache disabled and turbo disabled in real diagnostic calls.
- Do not run the full 14-file baseline until the final decision checkpoint.
- Do not commit or push.
- Do not touch the pre-existing deletions `entrada/.gitkeep` and `saida/.gitkeep`.

---

### Task 1: Preserve Existing Contracts

**Files:**
- Read: `tests/test_translation_engine.py`
- Read: `tests/test_subtitle_formats.py`

**Interfaces:**
- Consumes: `FixedASSTranslator.translate_single_batch(...)`, strict individual parser, segmented line-break recovery.
- Produces: a fresh green checkpoint before ITEM diagnostics.

- [ ] Run focused individual, line-break, selective-retry, residue, ID mapping, and batch parser tests.
- [ ] Confirm individual bare responses remain accepted only under strict single-line rules.
- [ ] Confirm missing/repositioned line breaks recover only after an eligible enveloped batch failure.
- [ ] Confirm a failed segment keeps the complete source item unchanged.

### Task 2: Build a Read-Only Real-Failure Corpus

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_extract.py`
- Create: `test_results/20260813_phase2/item_protocol_real_cases/cases.jsonl`
- Create: `test_results/20260813_phase2/item_protocol_real_cases/summary.json`
- Read: `test_results/20260813_current/baseline/trace.jsonl.gz`
- Read: `test_results/20260813_phase2/**/attempts.jsonl`

**Interfaces:**
- Consumes: trace events `batch_attempt_started`, `model_response`, `response_parsed`, `item_rejected`, and `item_failed`.
- Produces: one immutable record per malformed real response with source file, stable event ID when available, event index, expected ID, batch size, position, attempt, effective prompt, raw response, parsed IDs, parser warning, and final decision.

- [ ] Scan baseline and later experiment JSONL files without executing Ollama.
- [ ] Join each rejection to its exact prompt and preceding raw model response using file, mode, batch, attempt, and item IDs.
- [ ] Deduplicate identical prompt/response/expected-ID triples while preserving occurrence counts.
- [ ] Emit source hashes and artifact paths so every sample remains auditable.

### Task 3: Classify ITEM Deformations

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_classifier.py`
- Create: `test_results/20260813_phase2/item_protocol_real_cases/classified.jsonl`
- Modify: `test_results/20260813_phase2/item_protocol_real_cases/summary.json`

**Interfaces:**
- Consumes: raw response plus the literal expected IDs extracted from the effective prompt.
- Produces: zero or more evidence-backed labels from `ITEM_START_MISSING`, `ITEM_END_MISSING`, `ITEM_ID_MODIFIED`, `ITEM_DUPLICATED`, `ITEM_PARTIAL`, `ITEM_OUT_OF_ORDER`, `ITEM_TEXT_OUTSIDE_ENVELOPE`, `ITEM_FOREIGN_ID`, `ITEM_COUNT_MISMATCH`, and `ITEM_NESTED`.

- [ ] Tokenize every ITEM-like start/end marker independently from the production parser.
- [ ] Record literal start IDs, end IDs, paired IDs, physical order, duplicates, unknown IDs, residue spans, and nesting.
- [ ] Apply mutually documented precedence when one malformed response matches multiple labels.
- [ ] Manually inspect at least one raw response from every non-empty class.
- [ ] Publish counts by class, batch size, position, attempt, and source file.

### Task 4: Trace the First Divergence

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_real_cases/representative_trace.json`

**Interfaces:**
- Consumes: classified real cases and current `ITEM_BLOCK_RE` parser output.
- Produces: a representative sample showing prompt → raw Ollama response → marker scan → production parser → retry decision.

- [ ] Select cases covering every non-empty deformation class and the two known `batch_1_only` failures.
- [ ] Verify all expected envelopes are complete in the prompt before inference.
- [ ] Compare independent diagnostic scanning with the production parser result.
- [ ] Identify the first stage at which each case diverges.
- [ ] Separate model corruption, parser rejection, mapping rejection, and retry-only effects.

### Task 5: Controlled Real-Event Matrix

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_matrix.py`
- Create: `test_results/20260813_phase2/item_protocol_matrix/attempts.jsonl`
- Create: `test_results/20260813_phase2/item_protocol_matrix/summary.json`

**Interfaces:**
- Consumes: representative real events, `qwen2.5:14b`, current production prompts, temperature `0.2`, cache off, turbo off.
- Produces: repeated results for batch sizes 1, 2, 5, 10, and 15 at initial, middle, final, and original positions.

- [ ] Hold real neighbors fixed while moving one target event.
- [ ] Hold target position fixed while replacing neighbor groups.
- [ ] Repeat each relevant condition at least three times.
- [ ] Record envelope exactness, IDs, order, target mapping, response length, parser decision, retry count, individual recovery, and elapsed time.
- [ ] Classify each event as deterministic, position-dependent, batch-size-dependent, neighbor-dependent, and/or non-deterministic.

### Task 6: Compare Three One-ITEM Contracts

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_single_contracts.py`
- Create: `test_results/20260813_phase2/item_protocol_single_contracts/summary.json`

**Interfaces:**
- Consumes: the same real event and content under normal batch-1 prompt, selective retry prompt with the original ID, and strict individual recovery prompt.
- Produces: side-by-side prompt, temperature, parser, validation, raw response, and outcome evidence.

- [ ] Run batch with one ITEM at temperature `0.2`.
- [ ] Run selective retry with one ITEM at temperature `0.2` and its original non-1 ID.
- [ ] Run individual recovery at temperature `0.0`.
- [ ] Attribute outcome differences to prompt, ID, temperature, or parser only when controlled comparisons support the attribution.

### Task 7: Diagnostic Protocol and ID Alternatives

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_alternatives.py`
- Create: `test_results/20260813_phase2/item_protocol_alternatives/summary.json`

**Interfaces:**
- Consumes: identical real content and parameters with current envelopes plus structurally simpler diagnostic envelopes and numeric/alphabetic/opaque IDs.
- Produces: exact reproduction, omission, deformation, duplication, foreign-ID, and residue rates without modifying production.

- [ ] Compare current `<<<ITEM_0001>>> ... <<<END_ITEM_0001>>>` with at least two simpler paired-envelope formats.
- [ ] Compare IDs `0001`, `1`, `A`, and `X7K2` while keeping content and position identical.
- [ ] Repeat candidates in both one-item and multi-item contexts.
- [ ] Reject any candidate that improves reproduction by losing an unambiguous start/end ID pair.

### Task 8: Synthetic Cross-Contamination Controls

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_synthetic.py`
- Create: `test_results/20260813_phase2/item_protocol_synthetic/summary.json`

**Interfaces:**
- Consumes: highly distinguishable astronomy, cooking, and football text with unique literal anchors.
- Produces: ID-to-domain mapping evidence for correct, reordered, missing, duplicated, foreign, partial, nested, and outside-text responses.

- [ ] Verify the production parser maps valid reordered blocks strictly by ID.
- [ ] Verify duplicated IDs remain ambiguous and rejected.
- [ ] Verify missing or foreign IDs never borrow a body by order.
- [ ] Verify content outside complete envelopes is never assigned to an event.

### Task 9: Root-Cause and Architecture Checkpoint

**Files:**
- Modify: `docs/superpowers/plans/2026-08-13-item-batch-protocol-investigation.md`

**Interfaces:**
- Consumes: Tasks 2–8 evidence.
- Produces: one written causal hypothesis and one selected minimal production strategy with measured reliability, inference count, retry count, individual recovery count, elapsed-time ratio, and mapping risk.

- [ ] State the dominant deformation mechanism and the evidence that excludes prompt construction and parser corruption.
- [ ] Compare unchanged protocol, smaller fixed batches, automatic sub-batch splitting, binary recovery, one-call-per-ITEM recovery, and any causally supported envelope candidate.
- [ ] Select a strategy only if it preserves complete paired IDs and beats the current protocol on the same real sample.
- [ ] Add exact RED/GREEN implementation steps and production symbols to this plan before modifying production.

### Task 10: TDD Implementation After Confirmed Cause

**Files:**
- Modify: `tests/test_translation_engine.py`
- Modify only as justified: `translation_engine.py`
- Modify only as justified: `subtitle_formats.py`

**Interfaces:**
- Consumes: literal real malformed-response fixtures and the selected strategy from Task 9.
- Produces: a minimal recovery path that never maps an unidentified or conflicting body to an event.

- [ ] Write one focused test per confirmed causal behavior using literal expected translations and IDs.
- [ ] Run each new test against unchanged production and record the expected RED failure.
- [ ] Add negative tests for conflicting ID, duplicate ID, foreign ID, partial block, unidentified multiple bodies, external explanation, and unexpected Markdown.
- [ ] Implement only the selected recovery strategy.
- [ ] Run the focused tests and record GREEN.
- [ ] Mutation-check that removing ID validation, accepting duplicates, or assigning by order makes at least one test fail.

### Task 11: Directed and Full Verification

**Files:**
- Read: `tests/test_translation_engine.py`
- Read: `tests/test_subtitle_formats.py`
- Read: all `tests/`

**Interfaces:**
- Consumes: final implementation and unchanged prior corrections.
- Produces: fresh evidence for the delivery report.

- [ ] Run ITEM RED/GREEN tests.
- [ ] Run strict individual recovery tests.
- [ ] Run ASS line-break recovery, position validation, and atomicity tests.
- [ ] Run synthetic cross-contamination tests.
- [ ] Run the real ITEM sample before and after with identical parameters.
- [ ] Run `python -m pytest tests -q`.
- [ ] Run `git diff --check`.
- [ ] Report branch status without staging, committing, pushing, restoring `.gitkeep`, or including temporary artifacts.

### Task 12: Baseline Decision

**Files:**
- Create: `test_results/20260813_phase2/item_protocol_final_report.json`

**Interfaces:**
- Consumes: final real-sample metrics and full test evidence.
- Produces: explicit recommendation to run or defer the 14-file baseline.

- [ ] Report before/after counts for every ITEM deformation class.
- [ ] Report Ollama calls, retries, individual recoveries, and elapsed-time ratio.
- [ ] Confirm all accepted translations retained unambiguous ID-to-event association.
- [ ] Recommend a new full baseline only if no remaining objective failure group should be addressed first.

---

## Executed Evidence and Decision (2026-08-13)

### Regression checkpoint

- Initial focused checkpoint: `27 passed, 88 deselected`.
- Final strict individual checkpoint: `18 passed, 62 deselected`.
- Final ITEM/batch/synthetic checkpoint: `26 passed, 54 deselected`.
- Full suite: `166 passed in 3.21s`.

### Real corpus

- Baseline trace SHA-256: `09cd09aa50c95556e794aff460c4eb284cea32c1a3f034cd0aa9fe8e43818151`.
- Baseline malformed attempts: 36 occurrences / 33 unique responses.
- Baseline multi-label counts: outside text 31, count mismatch 31, partial 24,
  start missing 19, end missing 17, nested sequence 8, modified ID 3,
  duplicated ID 2. No foreign-ID deformation was observed.
- Every representative prompt contained all expected complete input envelopes. The first
  divergence was the raw Ollama response, before the production parser.
- The parser correctly retained valid complete IDs and rejected absent/duplicated IDs. A
  separate true-nesting fixture exposed one unsafe case in which the outer regex could absorb
  an inner ITEM; production now rejects that outer block.

### Controlled matrices

- Existing real event `evt_a65228cc6fb19c178773`: batch 1 failed the ITEM contract 3/3;
  batch retry with one ITEM failed 3/3; strict individual recovery succeeded 3/3 after the
  guarded bare-body correction.
- New real event `evt_a98156a2427be2763df2`: batch 1 failed 3/3; batch 2 succeeded 6/6;
  batch 5 succeeded 12/12; batch 10 failed 3/3 only in initial position and succeeded 9/9
  elsewhere; batch 15 initial was non-deterministic (2/3 accepted) and all other batch-15
  positions succeeded 9/9. Batch retry with one ITEM failed 3/3; strict individual recovery
  succeeded 3/3.
- Classification: `POSITION_DEPENDENT`, `BATCH_SIZE_DEPENDENT`, `NEIGHBOR_DEPENDENT`, and
  `NON_DETERMINISTIC`; batch-1 failures for the selected real events were deterministic.

### Protocol and identifier controls

- Thirty-six paired-envelope controls (current, angle, bracket; padded, plain, alphabetic,
  opaque IDs; batch sizes 1 and 3) reproduced exact pairs 36/36 under the explicit protocol
  instruction.
- Current paired envelopes with padded IDs and the explicit instruction reproduced exact
  pairs 3/3 on the original 15-item group.
- There is no causal evidence to change numeric IDs, remove zero padding, or replace the
  current paired delimiter format.

### Root cause

The model intermittently treats delimiters as presentational scaffolding rather than mandatory
data. Depending on content, neighbors, position, and response length it omits complete adjacent
pairs, emits bodies before their start marker, loses one or all `END_ITEM` markers, duplicates a
block, or edits a digit. The prompt is complete before inference, and the independent scanner
and production parser agree on the complete pairs actually returned. Therefore prompt
construction, parser loss, retry mapping, and numeric zero normalization are excluded as the
dominant source. Context-sensitive envelope reproduction by the model is the dominant mechanism.

### Selected minimum strategy

1. Strengthen only `_build_item_prompt()` with an explicit paired-envelope invariant.
2. Keep the existing strict ID parser and residue behavior; reject nested ITEM artifacts inside
   an otherwise complete outer body.
3. When a batch attempt already contains exactly one item and fails only because that ITEM is
   absent/duplicated, transition immediately to strict individual recovery at temperature 0.0
   instead of repeating the same unstable one-item batch contract.
4. Bump `PROMPT_VERSION` so cache keys cannot reuse translations from the old prompt contract.

No mapping by order, body count, or unidentified text was introduced.

### Before / after controlled sample

- Before, current prompt: batch 1 accepted 0/3 and batch 15 initial accepted 2/3; four of six
  initial calls had ITEM protocol failures. The exact baseline all-END-missing run consumed
  3 repeated full-batch calls plus 15 individual calls.
- After, full production workflow: 6/6 runs accepted (three batch 1 and three batch 15), with
  7 Ollama calls total, one selective retry, zero individual recoveries, and zero final failures.
- A separate direct-parser after sample had 5/6 exact initial responses; its remaining malformed
  batch-1 response was the exact class routed safely to individual recovery by the new rule.

### Baseline decision

The accumulated corrections now justify a new 14-file baseline: no known objective ITEM group
remains without a safe recovery path, the directed real sample is green, and the full automated
suite passes. The full baseline was intentionally not run in this investigation, per the staged
execution requirement; it should be the next independent validation action.
