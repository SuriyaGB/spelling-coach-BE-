# Spelling Coach Agent

TypeScript/Node backend for an AI-powered spelling coach agent.

## Requirements

- Node.js 20+ recommended
- npm
- An OpenAI API key for the default model/audio flow

## Install

```bash
npm install
```

## Local Run

Export the required environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Then start the API:

```bash
npm run dev
```

By default the server runs at:

```text
http://localhost:3000
```

You can change the port if needed:

```bash
export PORT=3001
npm run dev
```

## Environment Variables

### Required for default local usage

- `OPENAI_API_KEY`
  - Required for the default OpenAI model path and pronunciation audio.

### Optional model/runtime settings

- `SPELLING_COACH_MODEL`
  - Default: `openai:gpt-4.1-mini`
  - Examples:
    - `openai:gpt-4.1-mini`
    - `openai:gpt-5-mini`
    - `anthropic:claude-3-5-sonnet-latest`

- `SPELLING_COACH_RUNTIME`
  - Default: `deep_agent`
  - Allowed values:
    - `deep_agent`
    - `direct`

- `ANTHROPIC_API_KEY`
  - Required only if `SPELLING_COACH_MODEL` is set to an Anthropic model.

### Optional feature flags

- `SPELLING_COACH_AUDIO_CACHE`
  - Default: `on`
  - Set `off` to disable pronunciation audio caching.

- `SPELLING_COACH_TTS_INSTRUCTIONS`
  - Default: `off`
  - Set `on` to include explicit TTS instructions in the audio generation call.

- `SPELLING_COACH_RULE_SHORTLIST`
  - Default: `off`
  - Legacy spelling-rule shortlist flag.

- `SPELLING_COACH_RULE_PROMPT_HINTS`
  - Default: `off`
  - Legacy spelling-rule prompt-hints flag.

### Optional auth settings

- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`

These are only needed if you want authenticated routes to work, such as user-scoped custom lists.

## Common Local Start Commands

Default run:

```bash
export OPENAI_API_KEY="your_openai_api_key"
npm run dev
```

Run on port `3001` with audio cache disabled:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export PORT=3001
export SPELLING_COACH_AUDIO_CACHE=off
npm run dev
```

Run with direct runtime:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export SPELLING_COACH_RUNTIME=direct
npm run dev
```

## Tests

```bash
npm test
```

## Health Endpoint

Check server health and active feature flags:

```bash
curl http://localhost:3000/api/health
```

The response includes:

- active runtime
- active model
- feature flags such as audio caching and TTS instructions

## Data Storage

There is no database connection required for the current local backend.

Current persistence is file-based under `reference_data/`:

- `reference_data/words.generated.json`
- `reference_data/words.custom.generated.json`
- `reference_data/words.foreign.generated.json`

That means:

- regular word bank data is read from local JSON/reference files
- custom lists are currently stored locally in JSON
- foreign-origin word data is currently stored locally in JSON

## Supabase / DB Setup

There is currently no direct database connection string or ORM setup in this backend.

Supabase is used only for auth token verification on protected routes. The backend calls the Supabase Auth API using:

- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`

So for local backend usage:

- no PostgreSQL setup is required
- no migration step is required
- no DB connection env var is required

If you want to use authenticated custom-list flows from the UI, you should:

1. Create a Supabase project.
2. Enable the auth providers you want to use.
3. Set:
   - `SUPABASE_URL`
   - `SUPABASE_PUBLISHABLE_KEY`
4. Make sure the frontend signs users in and sends `Authorization: Bearer <access_token>` to protected endpoints.

## Protected vs Public Routes

Currently:

- Public routes:
  - regular practice flows
  - foreign-origin flows
  - health endpoint

- Auth-protected routes:
  - custom-list listing
  - custom-list detail
  - custom-list import
  - custom-list next-word lookup

## Notes

- The backend expects the reference data files in `reference_data/` to be present.
- The default model path uses OpenAI.
- Pronunciation audio also uses OpenAI TTS.
