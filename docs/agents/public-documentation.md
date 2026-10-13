## Public Documentation Guidelines

Public documentation, in doc/ and in Python's Docstring, and internal technical documentation, in docs/, serve fundamentally different purposes. **Never write public documentation as if it were an internal engineering report.**

### 1. Write for the intended audience

- Assume readers are users of the public API, not contributors or maintainers.
- Prioritize **what users can do, how to do it, and what results to expect**.
- Explain concepts using terminology familiar to the target audience.
- Do not assume knowledge of the project's architecture, implementation, or development history.
- Prefer concise, practical explanations and runnable examples over exhaustive technical discussions.

### 2. Respect abstraction boundaries

**Document each API at the abstraction level at which it is exposed.**

- Python API documentation must explain behavior in Python terms, not in terms of the underlying C++ implementation.
- Do not expose internal architecture, memory management strategies, implementation workarounds, or backend limitations unless they have a meaningful, observable consequence for users.
- When an implementation detail does affect users, explain the **user-visible behavior and how to handle it**, rather than the underlying engineering problem.
- Do not mix public API contracts with implementation notes.

For example, a Python user should not need to understand C++ object ownership or reference counting to use a Python function. If object lifetime matters, document the practical constraint using Python concepts.

### 3. Document the current state, not the history

**Public documentation describes how the software works now, not how it evolved.**

- Do not include descriptions of previous implementations, past bugs, internal refactorings, or the reasoning behind historical design changes.
- Avoid phrases such as "previously", "now supports", "was changed to", or "unlike the old implementation" unless explicitly writing a changelog or migration guide.
- When modifying existing documentation, integrate changes into the existing explanation instead of appending updates or corrections.
- Remove obsolete information rather than preserving it as historical context.
- Breaking changes, version differences, and migration instructions belong in dedicated documents.

### 4. Prefer task-oriented documentation

- Organize content around user goals, common workflows, and public API concepts rather than internal modules or implementation structure.
- Start with the simplest useful explanation; introduce complexity only when necessary.
- Prefer examples demonstrating typical usage over exhaustive descriptions of edge cases.
- Document important constraints, errors, and caveats, but only when they help users use the API correctly.
- Avoid overwhelming readers with defensive explanations of unlikely scenarios or internal failure modes.
- Keep advanced information separate from introductory explanations when possible.

### 5. Separate public and internal documentation

- **Public documentation:** tutorials, user guides, API references, examples, and user-visible behavioral constraints.
- **Internal documentation:** architecture, design decisions, implementation details, maintenance procedures, technical debt, and engineering tradeoffs.
- **Changelogs and migration guides:** historical changes, compatibility notes, and upgrade instructions.

Do not transfer content from one category to another without adapting it to the target audience.

### 6. Review documentation from the user's perspective

Before finalizing any public documentation change, verify:

1. **Audience:** Can someone unfamiliar with the implementation understand this?
2. **Relevance:** Does every explanation help the reader understand or use the public API?
3. **Abstraction:** Are internal implementation details leaking into user-facing explanations?
4. **Timelessness:** Does the text describe the current software rather than the process of changing it?
5. **Usability:** Can readers quickly find how to accomplish their task?
6. **Clarity:** Could any section be shorter, simpler, or better illustrated by an example?

**If a section reads like a developer handover, code review summary, or implementation report, rewrite it from the user's perspective rather than merely polishing its wording.**