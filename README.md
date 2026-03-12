# pg_hyperbolic

Native hyperbolic (Poincaré ball) distance for PostgreSQL. The first implementation of indexed approximate nearest-neighbor search in hyperbolic space in any database.

Built as a **companion extension** to [pgvector](https://github.com/pgvector/pgvector) — registers a custom operator class on pgvector's existing HNSW index without forking.

## Why Hyperbolic?

Euclidean embeddings waste dimensions encoding hierarchy. Hyperbolic space naturally represents trees and DAGs — entities near the origin are general, entities near the boundary are specific. A 32-dimensional Poincaré embedding can outperform a 768-dimensional Euclidean one for hierarchical data.

## Quick Start

```bash
# Build (requires PostgreSQL 16.x dev headers + pgvector 0.8.1+)
make && sudo make install
```

```sql
-- Enable
CREATE EXTENSION vector;           -- prerequisite
CREATE EXTENSION pg_hyperbolic;

-- Create table with vector column
CREATE TABLE items (
    id   serial PRIMARY KEY,
    name text,
    emb  vector(32)
);

-- Create HNSW index using Poincaré distance
CREATE INDEX ON items USING hnsw (emb vector_poincare_ops);

-- Query nearest neighbors
SELECT name, emb <~> '[0.1,0.2,...]'::vector AS dist
FROM items
ORDER BY emb <~> '[0.1,0.2,...]'::vector
LIMIT 10;
```

## Functions

### Core Distance & Utilities

| Function | Description |
|---|---|
| `poincare_distance(a, b)` | Poincaré ball distance: arcosh(1 + 2\|\|a-b\|\|² / ((1-\|\|a\|\|²)(1-\|\|b\|\|²))) |
| `a <~> b` | Operator form of `poincare_distance` |
| `poincare_norm(v)` | Euclidean norm \|\|v\|\| |
| `hyp_radius(v)` | Hyperbolic radius: 2·atanh(\|\|v\|\|) — encodes hierarchy depth |
| `hyp_project(v)` | Project onto open ball (clamp to \|\|v\|\| < 1 - ε) |
| `hyp_check_ball(v)` | Returns true if \|\|v\|\| < 1 |
| `hyp_conformal_factor(v)` | λ = 2/(1-\|\|v\|\|²) — metric scaling factor |
| `hyp_from_euclidean(v [, scale])` | Map Euclidean vector into Poincaré ball via exp₀ |

### Geometric Primitives

| Function | Description |
|---|---|
| `hyp_mobius_add(a, b)` | Möbius addition: a ⊕ b in the Poincaré ball |
| `hyp_midpoint(a, b)` | Hyperbolic midpoint via Einstein midpoint |
| `hyp_exp_map(base, tangent)` | Exponential map: tangent vector → ball point |
| `hyp_log_map(base, target)` | Logarithmic map: ball point → tangent vector |
| `hyp_lerp(a, b, t)` | Geodesic interpolation (t ∈ [0,1]) |
| `hyp_cone_contains(parent, child, K)` | Entailment cone test with aperture K |
| `hyp_busemann(point, direction)` | Busemann function (horocyclic height) |
| `hyp_parallel_transport(v, from, to)` | Parallel transport of tangent vector |
| `hyp_analogy(a, b, c)` | Hyperbolic analogy: "a is to b as c is to ?" |
| `hyp_dist_to_geodesic(p, a, b)` | Distance from point to geodesic through a,b |
| `hyp_ideal_direction(v)` | Normalize to unit sphere (ideal boundary) |

### Aggregate

| Function | Description |
|---|---|
| `hyp_centroid(v)` | Einstein midpoint aggregate — hyperbolic centroid via Klein model. Parallel-safe. |

### Variable Curvature (c > 0)

All standard operations parameterized by curvature c (ball radius = 1/√c):

`poincare_distance_c`, `hyp_mobius_add_c`, `hyp_radius_c`, `hyp_exp_map_c`, `hyp_log_map_c`, `hyp_project_c`, `hyp_lerp_c`

## HNSW Indexing

The `vector_poincare_ops` operator class plugs into pgvector's HNSW index. HNSW is metric-agnostic — it only compares distances, never does vector arithmetic — so Poincaré distance works without modification.

```sql
-- Standard HNSW parameters apply
CREATE INDEX ON items USING hnsw (emb vector_poincare_ops)
    WITH (m = 16, ef_construction = 200);

-- Set search beam width
SET hnsw.ef_search = 100;
```

**Benchmark results** (1,354 entities, 32D):
- Recall@10: 97.6%
- Indexed query: 0.59ms (vs 5.79ms brute-force)
- 60% neighbor divergence vs L2 — hyperbolic distance finds genuinely different neighbors

## Training Embeddings

Use the included training script to create Poincaré embeddings from a graph:

```bash
pip install gensim numpy psycopg2-binary

# From CSV edge list
python tools/train_poincare.py --edges edges.csv --dim 32 --dsn "postgresql://..." --table my_vectors

# Export to CSV only
python tools/train_poincare.py --edges edges.csv --dim 32 --output embeddings.csv
```

The edge list should have `source,target` columns representing hierarchical or relational edges.

## SQLite Implementation

A standalone SQLite loadable extension (`sqlite/hyp_sqlite.c`) implements the same core operations, proving the math is database-agnostic. Cross-validated to 12 decimal places against the PostgreSQL version.

```bash
cd sqlite && make
```

## Requirements

- PostgreSQL 16.x
- pgvector 0.8.1+
- C compiler (gcc/clang)

## Architecture

```
src/
  hyperbolic.h   — shared header: inline helpers, stability constants, SIMD hints
  hyperbolic.c   — core: distance, radius, projection, from_euclidean, conformal factor
  geometric.c    — Möbius add, exp/log maps, lerp, cones, Busemann, transport, analogy
  aggregate.c    — Einstein midpoint centroid (Klein model, parallel-safe)
  curvature.c    — variable curvature versions of all operations
sql/
  pg_hyperbolic--0.1.0.sql  — CREATE FUNCTION/OPERATOR/AGGREGATE statements
sqlite/
  hyp_sqlite.c   — standalone SQLite loadable extension
tools/
  train_poincare.py  — Gensim-based Poincaré embedding trainer
```

## Numerical Stability

- **Three-region arcosh**: zero-clamp (x ≤ 1) / Taylor series (x-1 < 1e-8) / standard formula — prevents catastrophic cancellation near identical points
- **Boundary clamping**: ε = 1e-7 (tighter than ML libraries)
- **Float32 storage, float64 intermediate computation**: compatible with pgvector's storage while maintaining precision

## License

Proprietary. Patent pending.
