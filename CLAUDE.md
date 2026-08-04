# AGENTS.md

## Coding style rules

- **YAGNI**: Only implement what is needed right now. No speculative features, no config options for hypothetical future use cases, no unused parameters.
- **KISS**: Prefer the simplest solution that works. Avoid clever tricks, extra layers of abstraction, or design patterns that aren't earning their keep.
- **Minimalist**: Fewest lines, fewest files, fewest new concepts. Don't create helpers/classes/modules for something used once.
- **Prefer one-liners**: When a one-liner is clear and correct, use it over a multi-line/multi-step version. Don't force it if it hurts readability, but default to compact.
- No comments unless explaining a non-obvious WHY (hidden constraint, workaround, surprising behavior). Never comment on WHAT the code does.
- No error handling/validation for cases that can't happen. Trust internal code; validate only at real boundaries (user input, external APIs).
- No backwards-compatibility shims, unused re-exports, or "just in case" fallbacks.
