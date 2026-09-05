https://iammdtanvirrahman2007.github.io/gAi/
# gAi

A human-guided, self-growing AI research project.

## Core principle

The AI learns concepts and identifies capabilities it needs, but it does **not write implementation code itself**. When a capability requires code, it creates a detailed request inside `code_requests/`. The human provides the implementation.

## Planned architecture

- `brain/` - core reasoning and learning logic
- `memory/` - short-term and long-term memory
- `knowledge/` - learned concepts
- `code_requests/` - requests for human-written implementations
- `experiments/` - controlled experiments
- `tests/` - verification

## Development rule

AI may propose and request code. Human approval and implementation are required for new executable capabilities.
