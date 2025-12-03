# Pro Architectural Designs

A collection of production-grade system design documents for complex, large-scale distributed systems. Unlike many existing system design resources that stay surface-level (or worse, contain mediocre content behind a paywall), these designs go deeper into architectural details, failure modes, and real-world trade-offs.

## Designs

| System | Description |
|--------|-------------|
| [Airbnb](./airbnb/DesignAirbnb.md) | End-to-end design for a stays marketplace covering search, booking, payments, messaging, and multi-region resilience |
| [Amazon Bedrock](./amazon-bedrock/DesignBedrock.md) | Serverless foundation model platform with inference orchestration, RAG, agents, and security hardening |

## What's Covered

Each design document includes:

- **Requirements & Scope** — Functional/non-functional requirements, scale targets, and SLOs
- **High-Level Architecture** — Component diagrams, API contracts, and data models
- **Deep Dives** — Detailed breakdowns of critical subsystems
- **Trade-offs & Reliability** — Bottleneck analysis, failure modes, and scaling strategies
- **Diagrams** — Mermaid.js architecture and sequence diagrams
