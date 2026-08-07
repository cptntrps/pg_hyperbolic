# pg_hyperbolic

## Project Overview
Native hyperbolic (Poincaré ball) distance extension for PostgreSQL, built as a companion to pgvector. First implementation of indexed approximate nearest-neighbor search in hyperbolic space in any database.

## Tech Stack
- C (PostgreSQL extension API, PGXS build system)
- PostgreSQL 16.x + pgvector 0.8.1
- SQLite (secondary implementation for patent breadth)
- Python (training script, benchmarks)

## Quick Start
```bash
# Build (inside Postgres environment with pg_config in PATH)
make && make install

# Enable
CREATE EXTENSION vector;  -- prerequisite
CREATE EXTENSION pg_hyperbolic;

# Create HNSW index
CREATE INDEX ON my_table USING hnsw (embedding vector_poincare_ops);

# Query
SELECT * FROM my_table ORDER BY embedding <~> query_vec LIMIT 10;
```

## Architecture
```
src/
  hyperbolic.c   — core distance, radius, projection, from_euclidean, conformal factor
  hyperbolic.h   — shared header: all inline helpers, stability constants, SIMD hints
  geometric.c    — Möbius add, exp/log maps, lerp, cones, Busemann, transport, analogy
  aggregate.c    — Einstein midpoint centroid (Klein model, parallel-safe)
  curvature.c    — variable curvature versions of all operations
sql/
  pg_hyperbolic--0.1.0.sql  — all CREATE FUNCTION/OPERATOR/AGGREGATE statements
sqlite/
  hyp_sqlite.c   — standalone SQLite loadable extension (patent breadth)
tools/
  train_poincare.py  — Gensim-based Poincaré embedding trainer
research/
  research reports (validation, internals, theory, operations, stability, benchmarks)
```

## Key Domain Concepts
- **Poincaré ball**: unit ball with d(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
- **Variable curvature**: c > 0 scales ball radius to 1/√c, all formulas parameterized
- **Companion extension pattern**: registers operator class on pgvector's HNSW without forking
- **Three-region arcosh**: zero-clamp / Taylor / standard for numerical stability
- **Hyperbolic radius**: 2·atanh(||x||) — encodes hierarchy depth, no Euclidean analog

## Constraints
- PostgreSQL 16.x required
- pgvector must be installed first
- Float32 storage with float64 intermediate computation
- Boundary clamping at eps=1e-7 (tighter than ML libraries)
- DO NOT push to GitHub until provisional patent is filed

## Current Phase
Complete (v0.3.0). 29 SQL functions, 1 operator, 1 HNSW operator class, 1 aggregate.
Patent provisional drafted, ready to file ($320 micro-entity).
