# Hyperbolic pgvector

## Project Overview
Native hyperbolic (Poincaré ball) distance extension for PostgreSQL — either extending pgvector or as a standalone extension.

## Tech Stack
- C (PostgreSQL extension API)
- PostgreSQL 16.x
- pgvector 0.8.1 (reference/dependency)
- Python (testing, benchmarks)
- geoopt (reference implementation for numerical patterns)

## Quick Start
- Build: `make` (inside Postgres container with pg_config in PATH)
- Install: `make install`
- Enable: `CREATE EXTENSION pg_hyperbolic;` (requires pgvector)
- Full implementation plan: `PLAN.md`

## Architecture
- `research/` — taskforce research reports (01-05)
- `src/` — C extension source (planned)
- `sql/` — SQL definitions, operator classes (planned)
- `test/` — regression tests + benchmark suite (planned)
- `bench/` — performance benchmarks against LivingOS data (planned)
- `livingos_schema_reference.sql` — schema snapshot from LivingOS
- `sample_*.csv` — sample data for testing

## Key Domain Concepts
- **Poincaré ball model**: unit ball with hyperbolic metric, d(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
- **Lorentz model**: hyperboloid alternative, numerically stabler, d(u,v) = arcosh(-<u,v>_L)
- **HNSW**: Hierarchical Navigable Small World graphs — the index structure to adapt
- **Entailment cones**: geometric hierarchy test — if B is in A's cone, A is more general
- **Möbius addition**: the "addition" operation in the Poincaré ball (non-commutative, non-associative)

## Constraints
- Must work with PostgreSQL 16.x
- Must handle boundary instability (norm → 1)
- Must support HNSW indexing with correct hyperbolic neighbor selection
- Real-world validation against LivingOS knowledge graph (9,463 nodes, 33,622 edges, 5,575 with 32D nav embeddings)

## Current Phase
Implementation — research complete, 5-phase build plan in PLAN.md. Phase 1 (core distance + HNSW) is next.
