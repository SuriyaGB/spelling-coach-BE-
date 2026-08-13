# Spelling Coach Architecture

## Overview

The spelling coach is split into two broad layers:

- offline or persisted word-level precompute
- runtime miss-specific coaching

The goal of the architecture is:

- keep stable word facts and teaching metadata out of the hot request path
- use the model at runtime only for child-attempt-specific coaching
- preserve deterministic backend control for chunking, pattern detection, pronunciation cues, and safety rules

At a high level:

1. word metadata is stored in JSON files under `reference_data/`
2. offline scripts enrich that metadata with chunking, phonemes, pronunciation tips, concept teaching, and related forms
3. runtime endpoints fetch the next word, warm cached word teaching if needed, and then respond to a spelling attempt
4. the runtime model focuses on miss analysis and coaching text, not form-teaching generation

## Core Data Files

Primary data files:

- `/Users/pavithra/spellingCoachAgent/reference_data/words.generated.json`
- `/Users/pavithra/spellingCoachAgent/reference_data/words.custom.generated.json`
- `/Users/pavithra/spellingCoachAgent/reference_data/words.foreign.generated.json`
- `/Users/pavithra/spellingCoachAgent/reference_data/pronunciation_overrides.json`
- `/Users/pavithra/spellingCoachAgent/reference_data/words.phonemes.generated.json`

Main persisted word fields in `words.generated.json`:

- core metadata
  - `word`
  - `definition`
  - `origin`
  - `part_of_speech`
  - `level`
  - `example_sentence`
- `word_breakdown`
  - `display_chunks`
  - `alternate_display_chunks`
  - `chunk_reason`
  - `matched_patterns`
- `phoneme_metadata`
  - `source`
  - `phonemes`
  - `sound_aware_patterns`
  - `friendly_chunks`
  - `say_aloud_tip`
  - `pronunciation_confidence`
- `word_teaching`
  - `concept_teaching`
- `concept_labels`

The lean phoneme export in `words.phonemes.generated.json` contains only:

- `word`
- `definition`
- `phoneme_metadata`

## Offline / Persisted Precompute

Offline precompute creates or enriches word-level metadata and writes it back into the JSON files.

### What Is Persisted Offline

Currently persisted:

- `word_breakdown`
  - chunking
  - alternate chunking
  - chunk reasoning
  - deterministic matched patterns
- `phoneme_metadata`
  - G2P phonemes
  - sound-aware matches
  - child-friendly pronunciation chunks
  - precomputed `say_aloud_tip`
  - pronunciation confidence
- `word_teaching.concept_teaching`
  - summary
  - meaning/origin/morphology focus
  - origin/morphology labels
  - related forms
- `concept_labels`

### Offline Scripts

Current scripts in `package.json`:

- `npm run precompute:breakdowns`
  - `/Users/pavithra/spellingCoachAgent/scripts/precomputeWordBreakdowns.ts`
- `npm run precompute:phonemes`
  - `/Users/pavithra/spellingCoachAgent/scripts/precomputeWordPhonemes.ts`
- `npm run precompute:g2p-pilot`
  - `/Users/pavithra/spellingCoachAgent/scripts/precomputeG2pPilot.ts`
- `npm run precompute:concepts`
  - `/Users/pavithra/spellingCoachAgent/scripts/precomputeConceptTeaching.ts`
- `npm run precompute:related-forms`
  - `/Users/pavithra/spellingCoachAgent/scripts/precomputeRelatedForms.ts`
- `npm run audit:pronunciations`
  - `/Users/pavithra/spellingCoachAgent/scripts/auditFriendlyPronunciations.ts`

### Script Responsibilities

`precompute:breakdowns`

- generates `word_breakdown`
- stores chunking metadata
- stores deterministic matched patterns

`precompute:phonemes`

- runs phoneme derivation
- computes sound-aware pattern metadata
- derives `friendly_chunks`
- derives stored `say_aloud_tip`
- sets pronunciation confidence

`precompute:concepts`

- generates concept-teaching content
- writes `word_teaching.concept_teaching`
- writes `concept_labels`

`precompute:related-forms`

- computes only `word_teaching.concept_teaching.related_forms`
- does not regenerate the rest of concept teaching
- currently scoped to Level 2 and Level 3 words
- supports checkpointing and retries

### Pronunciation Override Layer

Pronunciation exceptions are stored separately in:

- `/Users/pavithra/spellingCoachAgent/reference_data/pronunciation_overrides.json`

This layer exists so manual pronunciation fixes survive future recomputes.

Used for:

- corrected `friendly_chunks`
- corrected `say_aloud_tip`
- optional confidence adjustments

Runtime applies overrides through:

- `/Users/pavithra/spellingCoachAgent/app/pronunciationOverrides.ts`
- `/Users/pavithra/spellingCoachAgent/app/wordCatalog.ts`

## Runtime Responsibilities

Runtime should focus on the child’s actual attempt.

### What Runtime Still Does

At submit time, the model still generates:

- `missAnalysis`
- `errorRelevance`
- `teachingDecision`
- `coachingText.shortFeedback`
- `coachingText.fullExplanation`
- `coachingText.memoryTip`

Runtime also:

- merges stored word-level metadata into the response
- applies child-friendly `sayAloudTip`
- clears `fullExplanation` for correctly spelled words
- can suppress `nextStep` via feature flag
- interprets voice input and support requests

### What Runtime No Longer Should Generate

These are no longer intended to come from the runtime model:

- chunking
- matched pattern detection
- phoneme analysis
- pronunciation chunking
- form-teaching generation

Instead, runtime reads them from stored metadata:

- `wordBreakdown`
- `wordBreakdown.matchedPatterns`
- `coachingText.sayAloudTip`

## Runtime Data Flow

## 1. Next Word Fetch

Endpoint:

- `GET /api/words/next`

File:

- `/Users/pavithra/spellingCoachAgent/app/server.ts`

Behavior:

1. validates query params
2. picks a word from the correct source
   - main bank
   - custom list
   - foreign-origin list
3. returns a public word response quickly
4. kicks off a background `warmWordTeachingPrecompute(...)`

Important:

- `/api/words/next` still triggers a background warm step
- this is why logs can show:
  - `[spelling-coach precompute timing]`

That warm path lives in:

- `/Users/pavithra/spellingCoachAgent/app/optimizedCoach.ts`

## 2. Hear The Word

Endpoint:

- `GET /api/words/pronunciation?challengeId=<uuid>&sessionId=<uuid>`

File:

- `/Users/pavithra/spellingCoachAgent/app/server.ts`

Behavior:

- Resolves the target word from the active session via `challengeId` without exposing the word in the URL.
- Generates and returns pronunciation audio.
- Requires a valid bearer token (auth enforced via Supabase RLS).

## 3. Submit A Spelling Attempt

Endpoint:

- `POST /api/spelling-coach`

Files:

- `/Users/pavithra/spellingCoachAgent/app/server.ts`
- `/Users/pavithra/spellingCoachAgent/app/optimizedCoach.ts`
- `/Users/pavithra/spellingCoachAgent/app/runAgent.ts`

Behavior:

1. build normalized coaching input
2. decide whether to use split flow or full runtime flow
3. merge precomputed word-level metadata
4. run miss-specific model call
5. return final response

## Split Flow

The normal Level 2 and Level 3 path is the split flow.

Function:

- `runSplitSpellingCoachAgent(...)`

Behavior:

1. fetch or warm word-level precompute
2. make a smaller miss-only model call
3. merge:
   - stored/precomputed word data
   - runtime miss-only output

This is why timing logs show:

- `word_teaching_lookup`
- `miss_model_invoke_1`
- `merge_validation`

## Full Runtime Flow

The full runtime path is used when split-precompute is unavailable or not selected.

Function:

- `runSpellingCoachAgent(...)`

Behavior:

- still uses word-level precompute lookup
- but performs a larger runtime model step than the miss-only split path

## API Endpoints

Current important endpoints in `/Users/pavithra/spellingCoachAgent/app/server.ts`:

Health and status:

- `GET /api/health`

Word selection and pronunciation:

- `GET /api/words/next`
- `GET /api/words/pronunciation?challengeId=<uuid>&sessionId=<uuid>`

Coaching:

- `POST /api/spelling-coach`

Auth:

- `GET /api/auth/me`

Custom lists:

- `GET /api/custom-lists`
- `GET /api/custom-lists/:id`
- `POST /api/custom-lists/import`

Foreign-origin lists:

- `GET /api/foreign-origins`
- `GET /api/foreign-origins/:origin`
- `POST /api/foreign-origins/import`

Voice:

- `POST /api/voice/transcribe`
- `POST /api/voice/interpret`
- `POST /api/voice/respond`

## Feature Flags

Health endpoint returns active feature flags.

Important current flags:

- `SPELLING_COACH_NEXT_STEP`
  - when off, `nextStep` is returned empty
- `SPELLING_COACH_RUNTIME_CONCEPT_TEACHING`
  - controls whether runtime concept teaching generation is allowed
- spelling-rule matcher flags
- pronunciation/audio flags

Health output is assembled in:

- `/Users/pavithra/spellingCoachAgent/app/server.ts`

Prompt helpers and flag reads are in:

- `/Users/pavithra/spellingCoachAgent/app/prompt.ts`

## Word Sources

Words may come from three sources:

- main bank
  - `words.generated.json`
- custom lists
  - `words.custom.generated.json`
- foreign-origin lists
  - `words.foreign.generated.json`

Selection and normalization are handled in:

- `/Users/pavithra/spellingCoachAgent/app/wordCatalog.ts`

## Voice Safety Behavior

Voice interpretation is handled in:

- `/Users/pavithra/spellingCoachAgent/app/voice.ts`

Current safety behavior:

- if the child says exactly the target word, the backend blocks it
- if the child asks for help using the target word in the utterance, the display transcript masks it as `the challenge word`
- spoken letters remain valid as spelling attempts

## Current Known Architectural Reality

Not everything is fully offline-only yet.

Important nuance:

- a background warm precompute still runs on `/api/words/next`
- this can still do meaningful work if stored data is missing or if runtime concept-teaching generation is enabled

So the current architecture is best described as:

- mostly persisted word-level precompute
- plus a live warm-precompute layer
- plus runtime miss-only coaching

## Recommended Mental Model

Use this rule of thumb:

- if the data is a stable property of the word, it belongs offline
- if the data depends on what the child typed or said, it belongs at runtime

Offline:

- chunks
- matched patterns
- sound-aware matches
- pronunciation tips
- concept teaching
- related forms

Runtime:

- what mistake happened
- what to focus on now
- how to explain this attempt
- how to frame a memory tip for this attempt

## Files To Read First

If you want to understand the system quickly, start here:

- `/Users/pavithra/spellingCoachAgent/app/server.ts`
- `/Users/pavithra/spellingCoachAgent/app/optimizedCoach.ts`
- `/Users/pavithra/spellingCoachAgent/app/runAgent.ts`
- `/Users/pavithra/spellingCoachAgent/app/wordCatalog.ts`
- `/Users/pavithra/spellingCoachAgent/app/prompt.ts`
- `/Users/pavithra/spellingCoachAgent/app/schemas.ts`

Then for offline generation:

- `/Users/pavithra/spellingCoachAgent/scripts/precomputeWordBreakdowns.ts`
- `/Users/pavithra/spellingCoachAgent/scripts/precomputeWordPhonemes.ts`
- `/Users/pavithra/spellingCoachAgent/scripts/precomputeConceptTeaching.ts`
- `/Users/pavithra/spellingCoachAgent/scripts/precomputeRelatedForms.ts`

## Next Likely Cleanup Steps

Likely future simplifications:

- move concept teaching fully to persisted offline data with no live fallback needed
- reduce or eliminate expensive warm-precompute work on `/api/words/next` for already-persisted words
- keep runtime limited to miss-only coaching
- continue shifting stable word knowledge out of the runtime model path
