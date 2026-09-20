# Project guidance

Use the TypeSafe skill when working on this project. Read
[the installed skill](.agents/skills/typesafe-ai/SKILL.md) and follow its guidance.
Consult the live TypeSafe documentation before designing or changing a TypeSafe
integration, as the skill directs.

## Secrets

Never inspect or parse the project's `.env` file to expose its contents to the agent.
The user permits runtime credential loaders (such as `uv --env-file .env`) to read
it for API requests. This does not require another confirmation.
Never expose API keys in tool output, responses, logs, or tracked files.
Scripts must suppress secret-bearing debug output, exception text, and raw HTTP
console output. The user permits saving complete response bodies for debugging
after runtime credential redaction; never save authorization headers or cookies.
Exclude `.env` from file searches and content inspections.
