# Codebase Review

## Scope
This review covers the primary Neuro runtime sources (`src/`, `include/`), build setup (`Makefile`), and top-level usage docs (`README.md`).

## High-level assessment
- The project has a clear modular split between runtime orchestration (`AppController`), core engine state transitions/chemistry, UI modules, and llama runtime integration.
- The command-line flow and interactive shell in `src/main.cpp` are straightforward and easy to follow.
- The binary state persistence model is explicit and matches the operational intent of the runtime.

## Strengths
1. **Separation of concerns**
   - `AppController` provides a practical façade around locking, state lifecycle, and model calls.
2. **Defensive fallbacks for LLM output**
   - Decision parsing includes defaults and validation around state/mode fields.
3. **Reasonable runtime ergonomics**
   - Interactive commands support stepping, ticking chemistry, and knowledge ingestion with low overhead.

## Findings and recommendations

### 1) README command format is out of sync with runtime parser
- `README.md` documents `/knowledge-add <title>|<url>|<text>`.
- `src/main.cpp` currently parses `/knowledge-add <title>|<url>|<summary>|<text>`.

**Recommendation:** Align the README with actual parser behavior (or adjust parser behavior if 3-field input is intended).

### 2) Knowledge entry fallback can overwrite intent
- In `src/main.cpp`, when summary is empty, `entry.summary` is set to the full raw payload.

**Impact:** Inputs with an empty third segment can get an unexpectedly large summary payload.

**Recommendation:** Set fallback summary from `text` (if present) or from parsed components instead of the entire raw payload.

### 3) State save path has no surfaced error handling in interactive flow
- `app.save_state()` is invoked at process end; low-level failures surface as exceptions and terminate with a generic error line.

**Recommendation:** Consider explicit messaging around state write failures (path, permission, disk-full hints), while preserving current exception behavior.

### 4) Parsing strategy for model JSON is regex-based
- JSON field extraction uses regex; this is pragmatic but brittle under edge-case escaping or nested structures.

**Recommendation:** Move to a lightweight JSON parser when feasible to improve correctness and simplify maintenance.

## Suggested near-term priorities
1. Synchronize README `/knowledge-add` documentation with runtime behavior.
2. Tighten `/knowledge-add` fallback assignment logic.
3. Add clearer save-state failure diagnostics.
4. Plan a small migration from regex JSON extraction to a parser library.

## Summary
The runtime is in good shape for iterative development. Most review findings are maintainability and UX robustness improvements rather than structural defects.
