# Guessing or Asking? LLM listeners and child language trouble

## Study
At child turns containing trouble (unintelligible speech or a produced error), an LLM generates the next adult turn. We compare it with the real adult's reply: does the listener ask for clarification, reformulate the error correctly, or guess and move on? Target venue: ALIS 2027 (ACSW), full paper deadline 27 Nov 2026.

## Data
CHILDES Clinical-Eng (CHAT format) in data/raw/childes_clinical_eng/. Never modify raw files. data/raw/ is gitignored and must never be committed (TalkBank Ground Rules).

## Rules
- Transcripts may be sent ONLY to: OpenAI, Anthropic and Google Gemini paid APIs (keys from env vars OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, never written to files), and LM Studio at http://localhost:1234/v1. No other external service.
- Seed 42 everywhere. Processed data in data/processed/, model outputs in outputs/, analysis in results/.
- Every step ends with a markdown report in reports/ (what was done, counts, anything suspicious).
- Ask before changing the design. Flag problems instead of silently working around them.
