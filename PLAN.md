# pg_hyperbolic — Implementation Plan

**Date:** 2026-03-12
**Version:** 1.0
**Status:** Ready for implementation

---

## Table of Contents

1. [Architecture Design](#1-architecture-design)
2. [Implementation Roadmap](#2-implementation-roadmap)
3. [Risk Analysis & Decisions](#3-risk-analysis--decisions)

---

## 1. Architecture Design

### 1.1 Extension Structure

```
pg_hyperbolic/
├── Makefile                    # PGXS build system
├── pg_hyperbolic.control       # Extension metadata
├── src/
│   ├── hyperbolic.c            # Distance functions + PG wrappers (core)
│   ├── hyperbolic.h            # Shared constants, macros, inline helpers
│   ├── geometric.c             # Geometric primitives (midpoint, cone, transport, etc.)
│   └── aggregate.c             # Fréchet mean / Einstein midpoint aggregate
├── sql/
│   ├── pg_hyperbolic--0.1.0.sql  # CREATE FUNCTION/OPERATOR/OPCLASS statements
│   └── pg_hyperbolic--0.1.0--0.2.0.sql  # Upgrade path (future)
├── test/
│   ├── sql/                    # PostgreSQL regression tests
│   │   ├── distance.sql
│   │   ├── index_hnsw.sql
│   │   ├── index_ivfflat.sql
│   │   ├── geometric.sql
│   │   └── stability.sql
│   └── expected/               # Expected output for regression tests
│       ├── distance.out
│       ├── index_hnsw.out
│       ├── index_ivfflat.out
│       ├── geometric.out
│       └── stability.out
├── bench/
│   ├── bench_distance.c        # Raw C distance function benchmark
│   ├── bench_queries.sql       # SQL-level query benchmarks
│   └── bench_recall.py         # Recall@k measurement (indexed vs brute-force)
├── PLAN.md                     # This document
├── CLAUDE.md                   # Project instructions
├── VERSION                     # Semver
└── research/                   # Existing research reports (read-only reference)
```

**Total C files: 3** (hyperbolic.c ~250 lines, geometric.c ~250 lines, aggregate.c ~150 lines).
Each stays under 300 lines per project rules.

### 1.2 Build System

```makefile
# Makefile
EXTENSION = pg_hyperbolic
MODULE_big = pg_hyperbolic
OBJS = src/hyperbolic.o src/geometric.o src/aggregate.o
DATA = sql/pg_hyperbolic--0.1.0.sql
REGRESS = distance index_hnsw geometric stability

# pgvector header dependency
PG_CFLAGS += -I$(shell pg_config --includedir-server)/extension/vector

# Auto-vectorization flags (match pgvector's pattern)
PG_CFLAGS += -O2 -ftree-vectorize -fassociative-math \
             -fno-signed-zeros -fno-trapping-math

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
```

**pg_hyperbolic.control:**
```
comment = 'Hyperbolic (Poincaré) distance for pgvector with HNSW/IVFFlat indexing'
default_version = '0.1.0'
module_pathname = '$libdir/pg_hyperbolic'
requires = 'vector'
relocatable = true
```

The `requires = 'vector'` ensures pgvector is loaded first, making the `vector` type
and `hnsw`/`ivfflat` access methods available.

### 1.3 Data Flow

**INSERT path:**
```
User provides vector(N) in Poincaré coordinates
  → pg_hyperbolic validates: norm < 1 - eps (via checkValue in HnswTypeInfo)
  → If norm >= 1 - eps: radial projection to ball interior (direction-preserving)
  → Store as pgvector's float32 Vector (unchanged)
  → HNSW index calls poincare_distance() to compute neighbor distances
```

**QUERY path (ORDER BY ... <~> ...):**
```
Query vector arrives as vector(N)
  → HNSW search calls poincare_distance(query, candidate) via FunctionCall2Coll
  → poincare_distance():
      1. Upcast float32 components to float64
      2. Compute ||x||², ||y||², ||x-y||² in single loop
      3. Clamp norms: if n² >= 1 - 1e-7, set n² = 1 - 1e-7
      4. Compute arg = 1 + 2·||x-y||² / ((1-||x||²)(1-||y||²))
      5. Return stable_arcosh(arg)
  → HNSW uses return value for greedy search + neighbor pruning
  → Results returned as (vector, distance) pairs
```

**Decision: No Lorentz storage.** Convert on-the-fly during distance computation.
The Poincaré formula with double intermediates is precise enough (validated in Report 04),
and storing Lorentz would add a dimension (breaking user expectations for vector(32)).
The 2x speed advantage of Lorentz is real but premature — we optimize only after
the basic extension works. See Phase 4 for the Lorentz optimization path.

### 1.4 C Function Signatures

#### Core Distance (src/hyperbolic.c)

```c
/* --- Stability constants --- */
#define POINC_EPS           1e-7    /* boundary margin for norm clamping */
#define ARCOSH_TAYLOR_THR   1e-8    /* switch to Taylor series below this */

/* --- Internal computation functions --- */

/*
 * Stable arcosh for x >= 1.
 * Three regions: x<=1 → 0, x near 1 → Taylor, x>>1 → standard formula.
 */
static inline double
stable_arcosh(double x);

/*
 * Poincaré ball distance. float32 storage, float64 intermediates.
 * Decorated with VECTOR_TARGET_CLONES for auto-vectorization.
 */
VECTOR_TARGET_CLONES
static float
PoincareBallDistance(int dim, float *ax, float *bx);

/*
 * Radial projection: clamp vector to ball interior.
 * Preserves direction, scales norm to (1 - POINC_EPS).
 * Modifies vector in-place.
 */
static void
ProjectToBall(float *x, int dim);

/*
 * Hyperbolic radius: d(0, x) = 2 * atanh(||x||).
 */
static inline double
HyperbolicRadius(float *x, int dim);

/* --- PG-registered wrappers --- */

PG_FUNCTION_INFO_V1(poincare_distance);     /* (vector, vector) → float8 */
PG_FUNCTION_INFO_V1(poincare_norm);          /* (vector) → float8 (Euclidean norm) */
PG_FUNCTION_INFO_V1(hyp_radius);            /* (vector) → float8 (hyperbolic radius) */
PG_FUNCTION_INFO_V1(hyp_project);           /* (vector) → vector (project to ball) */
PG_FUNCTION_INFO_V1(hyp_check_ball);        /* (vector) → boolean (is inside ball?) */
PG_FUNCTION_INFO_V1(poincare_type_info);    /* () → internal (HnswTypeInfo pointer) */
```

#### Geometric Primitives (src/geometric.c)

```c
PG_FUNCTION_INFO_V1(hyp_midpoint);          /* (vector, vector) → vector */
PG_FUNCTION_INFO_V1(hyp_lerp);             /* (vector, vector, float8) → vector */
PG_FUNCTION_INFO_V1(hyp_cone_contains);    /* (vector, vector, float8) → boolean */
PG_FUNCTION_INFO_V1(hyp_busemann);         /* (vector, vector) → float8 */
PG_FUNCTION_INFO_V1(hyp_ideal_direction);  /* (vector) → vector */
PG_FUNCTION_INFO_V1(hyp_geodesic_walk);    /* (vector, vector, int) → SETOF vector */
PG_FUNCTION_INFO_V1(hyp_parallel_transport); /* (vector, vector, vector) → vector */
PG_FUNCTION_INFO_V1(hyp_analogy);          /* (vector, vector, vector) → vector */
PG_FUNCTION_INFO_V1(hyp_dist_to_geodesic); /* (vector, vector, vector) → float8 */
PG_FUNCTION_INFO_V1(hyp_mobius_add);       /* (vector, vector) → vector */
PG_FUNCTION_INFO_V1(hyp_exp_map);          /* (vector, vector) → vector (base, tangent → point) */
PG_FUNCTION_INFO_V1(hyp_log_map);          /* (vector, vector) → vector (base, point → tangent) */
```

#### Aggregate (src/aggregate.c)

```c
/*
 * Einstein midpoint aggregate via Klein model.
 * State: running sum of (gamma_i * x_klein_i) and sum of gamma_i,
 * plus dimension tracking.
 */
typedef struct HypCentroidState {
    int     dim;
    int     count;
    double  gamma_sum;          /* sum of Lorentz factors */
    double  weighted_sum[];     /* dim doubles: sum(gamma_i * k_i) */
} HypCentroidState;

PG_FUNCTION_INFO_V1(hyp_centroid_accum);    /* transition: (internal, vector) → internal */
PG_FUNCTION_INFO_V1(hyp_centroid_combine);  /* combine for parallel agg */
PG_FUNCTION_INFO_V1(hyp_centroid_final);    /* finalize: internal → vector */
PG_FUNCTION_INFO_V1(hyp_centroid_array);    /* non-aggregate: (vector[]) → vector */
```

### 1.5 SQL Operator and Operator Class Definitions

```sql
-- pg_hyperbolic--0.1.0.sql

-- ============================================================
-- Distance function
-- ============================================================
CREATE FUNCTION poincare_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================
-- Operator: <~> for Poincaré distance
-- ============================================================
-- NOTE: <~> is not used by pgvector (it uses <->, <=>, <#>, <+>).
-- We claim this operator symbol.
CREATE OPERATOR <~> (
    LEFTARG = vector, RIGHTARG = vector,
    PROCEDURE = poincare_distance,
    COMMUTATOR = '<~>'
);

-- ============================================================
-- HNSW Operator Class
-- ============================================================
-- Registers with pgvector's hnsw access method.
-- FUNCTION 1 = distance function (called by HNSW search/build).
-- FUNCTION 3 = type info (ball validation on insert).
CREATE OPERATOR CLASS vector_poincare_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <~> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector),
    FUNCTION 3 poincare_type_info(vector);

-- ============================================================
-- IVFFlat Operator Class (Phase 3)
-- ============================================================
-- FUNCTION 1 = scan distance
-- FUNCTION 3 = k-means distance (same as scan — Poincaré IS a metric)
-- FUNCTION 5 = type info (provides custom centroid computation)
-- CREATE OPERATOR CLASS vector_poincare_ops
--     FOR TYPE vector USING ivfflat AS
--     OPERATOR 1 <~> (vector, vector) FOR ORDER BY float_ops,
--     FUNCTION 1 poincare_distance(vector, vector),
--     FUNCTION 3 poincare_distance(vector, vector),
--     FUNCTION 5 poincare_ivf_type_info(vector);

-- ============================================================
-- Scalar functions
-- ============================================================
CREATE FUNCTION hyp_radius(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_project(vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_check_ball(vector) RETURNS boolean
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION poincare_norm(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================
-- Geometric primitives (Phase 2)
-- ============================================================
CREATE FUNCTION hyp_midpoint(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_lerp(vector, vector, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_cone_contains(vector, vector, float8 DEFAULT 0.1) RETURNS boolean
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_busemann(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_mobius_add(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_exp_map(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_log_map(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_analogy(vector, vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_dist_to_geodesic(vector, vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================
-- Aggregate (Phase 3)
-- ============================================================
CREATE FUNCTION hyp_centroid_accum(internal, vector) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION hyp_centroid_combine(internal, internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION hyp_centroid_final(internal) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE AGGREGATE hyp_centroid(vector) (
    SFUNC     = hyp_centroid_accum,
    STYPE     = internal,
    FINALFUNC = hyp_centroid_final,
    COMBINEFUNC = hyp_centroid_combine,
    PARALLEL  = SAFE
);
```

### 1.6 Type System Decision: Reuse pgvector's `vector`

**Decision: Reuse `vector`. Do NOT create a custom `hypvector` type.**

Rationale:
- The `vector` type is a float32 array with a dimension header. That is exactly what
  Poincaré coordinates are. No structural difference.
- Creating a new type would require duplicating pgvector's I/O functions, cast operators,
  TOAST handling, and all the plumbing that makes `vector` work.
- Users already have data in `vector` columns. A new type forces migration and casting.
- The operator class mechanism already separates "what type" from "what distance metric."
  That is the correct abstraction boundary — same type, different operator class.
- Ball membership validation happens at index build time via `checkValue` in the
  `HnswTypeInfo` struct (registered as FUNCTION 3 in the operator class). This is
  exactly how pgvector validates cosine vectors (checks norm != 0).

**Consequence:** A `vector(32)` column can have BOTH an L2 HNSW index and a Poincaré
HNSW index simultaneously. The query planner selects based on the operator used
(`<->` for L2, `<~>` for Poincaré). This is a feature, not a bug.

### 1.7 Lorentz Conversion Strategy

**Decision: On-the-fly conversion. No stored Lorentz representation in v0.x.**

The distance function computes Poincaré distance directly (not via Lorentz).
Rationale:
- Poincaré direct with double intermediates gives identical results to Lorentz
  (measured diff < 1e-15 in Report 04).
- Lorentz conversion requires allocating a (dim+1) temporary buffer per distance call.
  In HNSW search, this is called millions of times. The allocation overhead could
  negate the arithmetic savings.
- The Poincaré formula auto-vectorizes well with `target_clones` (the three-reduction
  loop is structurally identical to L2).
- If profiling shows the scalar tail (division + arcosh) is the bottleneck, we add
  Lorentz as an optimization in Phase 4 — pre-converting vectors at scan start and
  amortizing the conversion cost.

### 1.8 SIMD Strategy

Follow pgvector's `target_clones` pattern for the vector-phase loop:

```c
#ifdef __x86_64__
#define VECTOR_TARGET_CLONES \
    __attribute__((target_clones("avx512f,avx2,fma","avx","sse4.2","default")))
#elif defined(__aarch64__)
#define VECTOR_TARGET_CLONES \
    __attribute__((target_clones("sve","default")))
#else
#define VECTOR_TARGET_CLONES
#endif
```

The compiler generates multiple code paths and selects at runtime. The inner loop
of `PoincareBallDistance` has the same structure as `VectorL2SquaredDistance` — three
fused multiply-add reductions — so it auto-vectorizes equally well.

No hand-written intrinsics in v0.x. The scalar tail (division + arcosh) cannot benefit
from SIMD and is a fixed O(1) cost per distance call.

### 1.9 Error Handling

| Condition | Behavior |
|-----------|----------|
| Dimension mismatch (a.dim != b.dim) | `CheckDims(a, b)` — raises ERROR (pgvector provides this) |
| Vector outside ball (norm >= 1) | On index INSERT: radial projection to `1 - eps` with WARNING. On distance call: clamp norms silently (defensive, not destructive). |
| Zero vector (origin) | Valid Poincaré point (the "root"). Distance from origin = `2 * atanh(||y||)`. No special handling needed. |
| NaN/Inf in components | Raise ERROR on INSERT (pgvector's existing validation). Distance function returns NaN (propagates naturally). |
| arcosh argument < 1 | Return 0.0 (can happen only from floating-point imprecision when x ≈ y). |
| Very large distance (boundary points) | Clamped by norm eps. Maximum possible distance with eps=1e-7 is ~32 (finite, no overflow). |

---

## 2. Implementation Roadmap

### Phase 1: Core Distance — MVP (HNSW with Poincaré distance)

**Goal:** Replace the PL/pgSQL proof-of-concept with a C extension. Get HNSW indexing
working with correct Poincaré distance at native speed.

**Estimated time: 4-6 hours with AI assistance.**

#### Files to Create

**`Makefile`**
```makefile
EXTENSION = pg_hyperbolic
MODULE_big = pg_hyperbolic
OBJS = src/hyperbolic.o
DATA = sql/pg_hyperbolic--0.1.0.sql
REGRESS = distance index_hnsw

PG_CFLAGS += -I$(shell pg_config --includedir-server)/extension/vector
PG_CFLAGS += -O2 -ftree-vectorize -fassociative-math \
             -fno-signed-zeros -fno-trapping-math

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
```

**`pg_hyperbolic.control`**
```
comment = 'Hyperbolic (Poincaré) distance for pgvector HNSW/IVFFlat'
default_version = '0.1.0'
module_pathname = '$libdir/pg_hyperbolic'
requires = 'vector'
relocatable = true
```

**`src/hyperbolic.h`** (~60 lines)
```c
#ifndef PG_HYPERBOLIC_H
#define PG_HYPERBOLIC_H

#include "postgres.h"
#include "fmgr.h"
#include "extension/vector/vector.h"

/* Stability constants */
#define POINC_EPS           1e-7
#define ARCOSH_TAYLOR_THR   1e-8

/* Auto-vectorization (match pgvector pattern) */
#ifdef __x86_64__
#define HYP_TARGET_CLONES \
    __attribute__((target_clones("avx512f,avx2,fma","avx","sse4.2","default")))
#elif defined(__aarch64__)
#define HYP_TARGET_CLONES \
    __attribute__((target_clones("sve","default")))
#else
#define HYP_TARGET_CLONES
#endif

/* Stable arcosh */
static inline double
stable_arcosh(double x)
{
    if (x <= 1.0)
        return 0.0;
    double d = x - 1.0;
    if (d < ARCOSH_TAYLOR_THR)
        return sqrt(2.0 * d) * (1.0 - d / 12.0);
    return log(x + sqrt(x * x - 1.0));
}

/* Project vector to Poincaré ball interior (direction-preserving) */
static inline void
project_to_ball(float *x, int dim)
{
    double norm_sq = 0.0;
    for (int i = 0; i < dim; i++)
    {
        double xi = (double) x[i];
        norm_sq += xi * xi;
    }
    double max_norm = 1.0 - POINC_EPS;
    if (norm_sq > max_norm * max_norm)
    {
        double scale = max_norm / sqrt(norm_sq);
        for (int i = 0; i < dim; i++)
            x[i] = (float)((double) x[i] * scale);
    }
}

#endif /* PG_HYPERBOLIC_H */
```

**`src/hyperbolic.c`** (~180 lines)
```c
#include "hyperbolic.h"
#include <math.h>

PG_MODULE_MAGIC;

/* ---- Core distance computation ---- */

HYP_TARGET_CLONES
static float
PoincareBallDistance(int dim, float *ax, float *bx)
{
    double x_sq = 0.0, y_sq = 0.0, diff_sq = 0.0;

    for (int i = 0; i < dim; i++)
    {
        double xi = (double) ax[i];
        double yi = (double) bx[i];
        double d  = xi - yi;
        x_sq    += xi * xi;
        y_sq    += yi * yi;
        diff_sq += d * d;
    }

    /* Clamp norms to ball interior */
    if (x_sq >= 1.0 - POINC_EPS)
        x_sq = 1.0 - POINC_EPS;
    if (y_sq >= 1.0 - POINC_EPS)
        y_sq = 1.0 - POINC_EPS;

    double denom = (1.0 - x_sq) * (1.0 - y_sq);
    double arg   = 1.0 + 2.0 * diff_sq / denom;

    return (float) stable_arcosh(arg);
}

/* ---- PG-registered functions ---- */

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(poincare_distance);
Datum
poincare_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);

    CheckDims(a, b);

    PG_RETURN_FLOAT8((double) PoincareBallDistance(a->dim, a->x, b->x));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(poincare_norm);
Datum
poincare_norm(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  norm_sq = 0.0;

    for (int i = 0; i < v->dim; i++)
    {
        double xi = (double) v->x[i];
        norm_sq += xi * xi;
    }
    PG_RETURN_FLOAT8(sqrt(norm_sq));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_radius);
Datum
hyp_radius(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  norm_sq = 0.0;

    for (int i = 0; i < v->dim; i++)
    {
        double xi = (double) v->x[i];
        norm_sq += xi * xi;
    }
    double norm = sqrt(norm_sq);
    if (norm >= 1.0 - POINC_EPS)
        norm = 1.0 - POINC_EPS;

    /* d(0, x) = 2 * atanh(||x||) */
    PG_RETURN_FLOAT8(2.0 * atanh(norm));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_project);
Datum
hyp_project(PG_FUNCTION_ARGS)
{
    Vector *v    = PG_GETARG_VECTOR_P(0);
    Vector *result = InitVector(v->dim);

    memcpy(result->x, v->x, sizeof(float) * v->dim);
    project_to_ball(result->x, v->dim);

    PG_RETURN_VECTOR_P(result);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_check_ball);
Datum
hyp_check_ball(PG_FUNCTION_ARGS)
{
    Vector *v = PG_GETARG_VECTOR_P(0);
    double  norm_sq = 0.0;

    for (int i = 0; i < v->dim; i++)
    {
        double xi = (double) v->x[i];
        norm_sq += xi * xi;
    }
    PG_RETURN_BOOL(norm_sq < 1.0);
}
```

Note: `CheckDims` is provided by pgvector's `vector.h` — it raises an ERROR if
dimensions mismatch. `InitVector` allocates a new Vector with the given dimension.

**`sql/pg_hyperbolic--0.1.0.sql`** (Phase 1 subset — only distance + HNSW)
```sql
-- Poincaré distance
CREATE FUNCTION poincare_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Operator
CREATE OPERATOR <~> (
    LEFTARG = vector, RIGHTARG = vector,
    PROCEDURE = poincare_distance,
    COMMUTATOR = '<~>'
);

-- HNSW operator class
CREATE OPERATOR CLASS vector_poincare_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <~> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector);

-- Utility functions
CREATE FUNCTION poincare_norm(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_radius(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_project(vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_check_ball(vector) RETURNS boolean
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
```

#### Tests (Phase 1)

**`test/sql/distance.sql`**
```sql
-- Basic distance properties
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_hyperbolic;

-- Identity: d(x,x) = 0
SELECT poincare_distance('[0.1,0.2,0.3]'::vector, '[0.1,0.2,0.3]'::vector);
-- Expected: 0

-- Symmetry: d(x,y) = d(y,x)
SELECT poincare_distance('[0.1,0.2,0.3]'::vector, '[0.4,0.3,0.1]'::vector)
     = poincare_distance('[0.4,0.3,0.1]'::vector, '[0.1,0.2,0.3]'::vector);
-- Expected: true

-- Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
SELECT poincare_distance('[0.1,0.2]'::vector, '[0.5,0.3]'::vector)
    <= poincare_distance('[0.1,0.2]'::vector, '[0.3,0.1]'::vector)
     + poincare_distance('[0.3,0.1]'::vector, '[0.5,0.3]'::vector);
-- Expected: true

-- Distance from origin grows as norm increases
SELECT hyp_radius('[0.1,0,0]'::vector) < hyp_radius('[0.5,0,0]'::vector);
-- Expected: true

-- Near-boundary stability: should not be NaN or Inf
SELECT poincare_distance('[0.999,0,0]'::vector, '[0,0.999,0]'::vector);
-- Expected: finite positive number (~7.6)

-- Dimension mismatch raises error
SELECT poincare_distance('[0.1,0.2]'::vector, '[0.1,0.2,0.3]'::vector);
-- Expected: ERROR

-- Ball membership check
SELECT hyp_check_ball('[0.5,0.3,0.1]'::vector);  -- true
SELECT hyp_check_ball('[0.9,0.9,0.9]'::vector);  -- false (norm > 1)

-- Projection
SELECT poincare_norm(hyp_project('[0.9,0.9,0.9]'::vector)) < 1.0;
-- Expected: true
```

**`test/sql/index_hnsw.sql`**
```sql
-- HNSW index with Poincaré distance
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_hyperbolic;

-- Create test table with Poincaré vectors
CREATE TABLE test_poincare (id serial PRIMARY KEY, v vector(3));
INSERT INTO test_poincare (v) VALUES
    ('[0.1,0.2,0.3]'), ('[0.4,0.1,0.2]'), ('[0.3,0.3,0.1]'),
    ('[0.5,0.2,0.1]'), ('[0.1,0.1,0.1]'), ('[0.2,0.4,0.3]'),
    ('[0.6,0.1,0.1]'), ('[0.1,0.5,0.2]'), ('[0.3,0.2,0.4]'),
    ('[0.2,0.2,0.2]');

-- Build HNSW index
CREATE INDEX idx_poincare_hnsw ON test_poincare
    USING hnsw (v vector_poincare_ops);

-- KNN query should use index
SET enable_seqscan = off;
SELECT id, v <~> '[0.2,0.2,0.2]'::vector AS dist
FROM test_poincare
ORDER BY v <~> '[0.2,0.2,0.2]'::vector
LIMIT 5;

-- Verify index was used (check plan)
EXPLAIN (COSTS OFF)
SELECT id FROM test_poincare
ORDER BY v <~> '[0.2,0.2,0.2]'::vector
LIMIT 5;
-- Expected: Index Scan using idx_poincare_hnsw

DROP TABLE test_poincare;
```

**`test/sql/stability.sql`**
```sql
-- Boundary stability tests
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_hyperbolic;

-- Near-boundary points should produce finite distances
SELECT poincare_distance(
    '[0.999,0.001,0.001]'::vector,
    '[0.001,0.999,0.001]'::vector
) BETWEEN 0 AND 100;
-- Expected: true

-- Very close points should produce small distances
SELECT poincare_distance(
    '[0.5,0.3,0.1]'::vector,
    '[0.500001,0.300001,0.100001]'::vector
) < 0.001;
-- Expected: true

-- Origin to any point should equal 2*atanh(norm)
SELECT round(poincare_distance('[0,0,0]'::vector, '[0.5,0,0]'::vector)::numeric, 6)
     = round((2 * atanh(0.5))::numeric, 6);
-- Expected: true
```

#### Dependencies

- PostgreSQL 16 development headers (`postgresql-server-dev-16`)
- pgvector 0.8.1 installed (provides `vector` type, `hnsw` AM, header files)

#### Build & Install (in Docker container)

```bash
# Copy source into container
docker cp src/ livingos-postgres:/tmp/pg_hyperbolic/src/
docker cp sql/ livingos-postgres:/tmp/pg_hyperbolic/sql/
docker cp Makefile livingos-postgres:/tmp/pg_hyperbolic/
docker cp pg_hyperbolic.control livingos-postgres:/tmp/pg_hyperbolic/

# Build inside container
docker exec livingos-postgres bash -c "cd /tmp/pg_hyperbolic && make && make install"

# Create extension
PGPASSWORD=livingos2026 psql -U livingos -d livingos -p 5435 -h localhost \
    -c "CREATE EXTENSION pg_hyperbolic;"
```

---

### Phase 2: Geometric Primitives

**Goal:** Add Möbius addition, midpoint, cone containment, interpolation, and manifold maps.

**Estimated time: 3-4 hours with AI assistance.**

**Depends on: Phase 1 (core distance must be working and tested).**

#### Files to Create/Modify

**`src/geometric.c`** (~250 lines) — new file

C functions to implement:

| Function | Signature | Complexity | Notes |
|----------|-----------|------------|-------|
| `hyp_mobius_add` | `(vector, vector) → vector` | O(n) | Core building block. Möbius addition with denominator clamping. |
| `hyp_midpoint` | `(vector, vector) → vector` | O(n) | Via conformal factors: `(λ²_y·x + λ²_x·y) / (λ²_y + λ²_x - 2 + eps)`. Project result to ball. |
| `hyp_lerp` | `(vector, vector, float8) → vector` | O(n) | Geodesic interpolation at parameter t. Uses exp/log maps. |
| `hyp_cone_contains` | `(vector, vector, float8) → bool` | O(n) | Simplified test: angle between x and `(-x)⊕y` vs `psi(x) = arcsin(K·(1-‖x‖²)/‖x‖)`. |
| `hyp_busemann` | `(vector, vector) → float8` | O(n) | `ln((1-‖x‖²) / ‖x-θ‖²)`. Theta normalized to unit vector internally. |
| `hyp_exp_map` | `(vector, vector) → vector` | O(n) | `exp_x(v) = x ⊕ tanh(λ_x·‖v‖/2)·v/‖v‖`. |
| `hyp_log_map` | `(vector, vector) → vector` | O(n) | `log_x(y) = (2/(λ_x·α))·atanh(α)·u` where `u = (-x)⊕y, α = ‖u‖`. |
| `hyp_parallel_transport` | `(vector, vector, vector) → vector` | O(n) | Transport tangent v from T_x to T_y. Uses `(λ_x/λ_y)·gyr[y,-x](v)`. |
| `hyp_analogy` | `(vector, vector, vector) → vector` | O(n) | Composition: `exp_y(PT_{a→y}(log_a(b)))`. |
| `hyp_dist_to_geodesic` | `(vector, vector, vector) → float8` | O(n) | Hyperbolic law of cosines: `arcsinh(|sinh(d(a,x))·sin(angle_a)|)`. |
| `hyp_ideal_direction` | `(vector) → vector` | O(n) | `x / ‖x‖` (normalize to unit sphere). |
| `hyp_geodesic_walk` | `(vector, vector, int) → SETOF vector` | O(steps·n) | SRF using `hyp_lerp` at evenly spaced t values. |

**`sql/pg_hyperbolic--0.1.0.sql`** — add geometric function registrations (listed in 1.5 above).

**`Makefile`** — add `src/geometric.o` to OBJS.

#### Tests

**`test/sql/geometric.sql`**
- Midpoint equidistance: `d(a, mid) ≈ d(b, mid)`
- Midpoint is inside ball: `hyp_check_ball(hyp_midpoint(a, b))`
- Cone containment: origin contains everything, boundary contains nothing
- Cone asymmetry: `a @> b` does not imply `b @> a`
- Möbius addition: `a ⊕ 0 = a` (identity element is origin)
- Exp/log roundtrip: `exp_x(log_x(y)) ≈ y`
- Lerp endpoints: `hyp_lerp(a, b, 0) ≈ a`, `hyp_lerp(a, b, 1) ≈ b`
- Busemann sign: points in direction θ have positive Busemann value

---

### Phase 3: IVFFlat Support & Aggregate

**Goal:** Enable IVFFlat indexing with correct hyperbolic centroids. Add `hyp_centroid` aggregate.

**Estimated time: 3-4 hours with AI assistance.**

**Depends on: Phase 1 (distance), Phase 2 (midpoint, for validation).**

#### Files to Create/Modify

**`src/aggregate.c`** (~150 lines) — new file

The `hyp_centroid` aggregate uses the Einstein midpoint via the Klein model:

```
Transition (per row):
  1. Poincaré→Klein: k_i = 2·x_i / (1 + ‖x_i‖²)
  2. Lorentz factor: γ_i = 1 / √(1 - ‖k_i‖²)
  3. Accumulate: weighted_sum += γ_i · k_i,  gamma_sum += γ_i

Finalize:
  1. k_mu = weighted_sum / gamma_sum
  2. Klein→Poincaré: mu = k_mu / (1 + √(1 - ‖k_mu‖²))
  3. Project to ball (safety)
```

State is `HypCentroidState`: dim (int) + count (int) + gamma_sum (double) + weighted_sum (double[dim]).

The `COMBINEFUNC` enables parallel aggregation by adding the weighted sums and gamma sums
from two partial states.

**IVFFlat integration:**

pgvector's IVFFlat looks up `IVFFLAT_TYPE_INFO_PROC` (FUNCTION 5) to get a custom
`IvfflatTypeInfo` struct. We provide `poincare_ivf_type_info` which returns a struct with:
- `sumCenter`: accumulate Klein-projected Lorentz-weighted sums
- `updateCenter`: finalize by dividing by gamma sum and converting Klein→Poincaré

This requires understanding pgvector's internal `IvfflatTypeInfo` struct layout.
We access it by including pgvector's headers. If the struct is not in the public header,
we replicate its definition (it's stable across pgvector minor versions).

**`sql/pg_hyperbolic--0.1.0.sql`** — add:
- `hyp_centroid` aggregate registration
- IVFFlat operator class registration (uncomment from Section 1.5)

**`Makefile`** — add `src/aggregate.o` to OBJS. Add `index_ivfflat` to REGRESS.

#### Tests

**`test/sql/index_ivfflat.sql`**
- IVFFlat index creation with `vector_poincare_ops`
- KNN query uses IVFFlat index (EXPLAIN check)
- Results match brute-force ordering

**Centroid tests (in geometric.sql or separate):**
- `hyp_centroid` of a single point returns that point
- `hyp_centroid` is inside the ball
- `hyp_centroid` of symmetric points around origin ≈ origin
- Parallel aggregate produces same result as serial

---

### Phase 4: Optimization

**Goal:** Benchmark, profile, and optimize the hot path. Add Lorentz computation option.

**Estimated time: 2-3 hours with AI assistance.**

**Depends on: Phases 1-3 (everything working and tested).**

#### Benchmarks to Create

**`bench/bench_distance.c`** — standalone C program (not a PG extension):
- Measures raw throughput: Poincaré vs L2 vs Lorentz at 32D and 768D
- Uses the same vector data from `sample_nav_vectors.csv`
- Reports ops/sec and latency percentiles

**`bench/bench_queries.sql`** — SQL benchmarks:
- Index build time: HNSW with `vector_poincare_ops` vs `vector_l2_ops`
- KNN query latency at k=1,5,10,20,50
- Recall@k measurement: indexed vs sequential scan

**`bench/bench_recall.py`** — Python script:
- Loads sample data into PG
- Runs brute-force KNN (sequential scan)
- Runs indexed KNN (HNSW)
- Computes recall@k for k=1,5,10
- Reports results as markdown table

#### Optimization Targets

1. **Lorentz fast path:** If benchmarks show the scalar tail (division + arcosh) accounts
   for >20% of distance computation time, implement a Lorentz-based distance function.
   Strategy: at scan start, convert the query vector to Lorentz (once). For each candidate,
   convert to Lorentz inline (fused with the inner product loop — no separate allocation).

2. **Batch conversion:** If HNSW search shows per-call FunctionCall2Coll overhead is
   significant, investigate pgvector's batch distance API (if any). Currently pgvector
   calls distance one pair at a time.

3. **Memory layout:** Ensure the inner loop accesses vectors sequentially (no pointer
   chasing). pgvector's `Vector.x` is already contiguous float32 — optimal for cache.

#### Files to Modify

- `src/hyperbolic.c` — potentially add `LorentzDistance` function and a `poincare_distance_lorentz` PG wrapper
- `sql/pg_hyperbolic--0.1.0.sql` — add Lorentz distance function registration (user-selectable)

---

### Phase 5: Advanced Features

**Goal:** Radius-aware HNSW layer assignment, variable curvature, and extended operations.

**Estimated time: 4-6 hours with AI assistance.**

**Depends on: Phase 4 (benchmarks confirming baseline performance).**

#### 5a. Variable Curvature

Add a curvature parameter `c` (default 1.0) to all functions. The generalized Poincaré
distance is:

```
d_c(u,v) = (2/√c) · arctanh(√c · ‖(-u)⊕_c v‖)
```

Implementation: add `float8 c` as an optional last parameter to each function.
Store curvature as a reloption on the index (via `CREATE INDEX ... WITH (curvature = 0.5)`).

**Decision: Defer to Phase 5.** All LivingOS embeddings use c=1. Variable curvature
adds complexity to every function without immediate value. When users request it (or
when we need it for mixed-curvature product spaces), add it then.

#### 5b. Radius-Aware Layer Assignment

Prokhorenkova et al. (2022) suggest that HNSW in hyperbolic space may benefit from
assigning higher layers to points near the origin (natural hubs). This requires
modifying HNSW's layer assignment formula:

```
standard:  layer = floor(-ln(uniform()) * m_L)
modified:  layer = floor(-ln(uniform()) * m_L * (1 + alpha * (1 - ||x||)))
```

This is NOT possible via the operator class mechanism — it requires modifying
pgvector's HNSW insertion code. Options:
1. Fork pgvector and add a hook for custom layer assignment
2. Propose a patch upstream (pgvector issue for configurable layer assignment)
3. Accept standard layer assignment (which already works — validated at 99.5% recall)

**Decision: Accept standard layer assignment.** 99.5% recall is already excellent.
Radius-aware assignment is a research direction, not a production requirement.

#### 5c. Additional Operations (if demand exists)

- `hyp_variance(vector) → float8` — Fréchet variance (sum of squared distances to centroid)
- `hyp_diameter(vector[]) → float8` — max pairwise distance in a set
- `hyp_horosphere_project(vector, vector) → vector` — project onto horosphere at ideal point

---

## 3. Risk Analysis & Decisions

### 3.1 float32 vs float64 Storage

**Decision: float32 (pgvector's `vector` type).**

Justification:
- pgvector's `Vector` struct uses `float x[]` (32-bit). Reusing this type is mandatory
  for the companion extension approach — we cannot change pgvector's storage format.
- Report 04 measured: float32 storage with float64 intermediates gives relative error
  < 4.1e-5 even at norm=0.9999. This is more than sufficient for nearest-neighbor
  ordering (we need correct ranking, not exact distances).
- 2x storage savings (128 bytes vs 256 bytes for 32D; 3KB vs 6KB for 768D).
- SIMD processes 8 float32 values per AVX-256 register (vs 4 float64).
- The training process itself introduces far more error than float32 quantization.
- `halfvec` (float16) support via pgvector is an option for even more compression,
  but Poincaré distance near the boundary would lose too much precision. Not recommended.

### 3.2 Lorentz Internal Storage vs On-the-Fly Conversion

**Decision: On-the-fly conversion (no Lorentz storage).**

Justification:
- Lorentz storage adds 1 dimension per vector (33 instead of 32). This changes the
  user-visible vector dimensionality, breaking the mental model.
- Conversion cost is 77ns per vector (measured). For a typical HNSW search scanning
  ~200 candidates, that's 15μs — negligible vs total query time.
- BUT: we don't even need conversion in v0.x. The Poincaré formula with double
  intermediates is precise and reasonably fast (0.46x L2 throughput, auto-vectorized).
  The arcosh overhead converges to 0 at higher dimensions (768D: ~1.1x L2).
- If profiling shows conversion is worthwhile, add it in Phase 4 as an inline
  transformation fused with the inner product loop (no separate allocation).

### 3.3 Variable Curvature from Day 1

**Decision: No. Fixed c=1 for v0.x.**

Justification:
- All LivingOS Poincaré embeddings use c=1 (standard Poincaré ball).
- geoopt uses c=1 by default; most academic papers use c=1.
- Adding curvature to every function doubles the parameter surface and testing matrix.
- The curvature parameter can be added as a backward-compatible optional argument
  later: `poincare_distance(a, b)` (c=1) vs `poincare_distance(a, b, 0.5)`.
- The SQL operator `<~>` is inherently c=1. Variable curvature would need a function
  call form, not an operator.

### 3.4 Licensing

**Decision: PostgreSQL License (same as pgvector).**

Justification:
- pgvector uses the PostgreSQL license (essentially BSD/MIT).
- Using the same license eliminates any compatibility concerns.
- Maximizes adoption — no copyleft concerns for enterprise users.
- MIT and Apache 2.0 are also fine, but matching pgvector reduces friction.

### 3.5 Name

**Decision: `pg_hyperbolic`.**

Rationale:
- `pg_poincare` — too specific. We may add Lorentz storage, variable curvature,
  or even spherical distance later. "Hyperbolic" covers the mathematical domain.
- `hypervector` — confusing. "Hypervector" is an existing term in computing
  (hyperdimensional computing / VSA) that means something completely different.
- `pg_hyperbolic` — clear, accurate, follows the `pg_` prefix convention for
  PostgreSQL extensions. Describes what it does (hyperbolic geometry operations)
  without overcommitting to a specific model.

Extension name in SQL: `CREATE EXTENSION pg_hyperbolic;`
Shared library: `pg_hyperbolic.so`
Control file: `pg_hyperbolic.control`

### 3.6 Test Strategy

**Three tiers:**

#### Tier 1: PostgreSQL Regression Tests (`make installcheck`)
- Located in `test/sql/` with expected output in `test/expected/`
- Run by `make installcheck` (standard PGXS testing)
- Cover: distance properties (identity, symmetry, triangle inequality), index creation,
  KNN queries, boundary stability, error handling, geometric primitive correctness
- **Must pass before any release.**

#### Tier 2: Benchmark Suite (`bench/`)
- `bench_distance.c` — raw C throughput (not PG overhead)
- `bench_queries.sql` — SQL-level latency and QPS
- `bench_recall.py` — recall@k accuracy measurement
- Run manually, results tracked in git (committed as markdown tables)
- **Must show:** recall@10 >= 99%, throughput >= 0.3x L2 (auto-vectorized)

#### Tier 3: Integration Tests (LivingOS)
- Load real LivingOS nav embeddings (5,575 nodes, 32D) into `test_hyperbolic` table
- Build HNSW index with `vector_poincare_ops`
- Run known queries ("Python" neighbors, cone containment, midpoint bridging)
- Compare results against the validated PL/pgSQL baseline
- **Must show:** identical top-10 neighbor ordering to the PL/pgSQL proof-of-concept

### 3.7 Migration Path for LivingOS

**Current state:**
- `graph.nodes.poincare_nav` — `double precision[]` (32D), 5,575 nodes
- `graph.nodes.poincare_embedding` — `double precision[]` (768D), 5,873 nodes
- No indexes on these columns (sequential scan for all KNN)

**Migration steps:**

#### Step 1: Add vector columns (non-destructive)
```sql
ALTER TABLE graph.nodes
    ADD COLUMN poincare_nav_vec vector(32),
    ADD COLUMN poincare_emb_vec vector(768);
```

#### Step 2: Populate from existing arrays
```sql
UPDATE graph.nodes
SET poincare_nav_vec = ('[' || array_to_string(poincare_nav, ',') || ']')::vector(32)
WHERE poincare_nav IS NOT NULL;

UPDATE graph.nodes
SET poincare_emb_vec = ('[' || array_to_string(poincare_embedding, ',') || ']')::vector(768)
WHERE poincare_embedding IS NOT NULL;
```

#### Step 3: Project to ball (ensure all vectors are valid)
```sql
UPDATE graph.nodes
SET poincare_nav_vec = hyp_project(poincare_nav_vec)
WHERE poincare_nav_vec IS NOT NULL
  AND NOT hyp_check_ball(poincare_nav_vec);
```

#### Step 4: Build HNSW indexes
```sql
-- 32D nav embeddings — fast to build (~seconds for 5,575 vectors)
CREATE INDEX idx_nodes_poincare_nav_hnsw
ON graph.nodes USING hnsw (poincare_nav_vec vector_poincare_ops)
WITH (m = 16, ef_construction = 200);

-- 768D full embeddings — slower to build (~minutes for 5,873 vectors)
CREATE INDEX idx_nodes_poincare_emb_hnsw
ON graph.nodes USING hnsw (poincare_emb_vec vector_poincare_ops)
WITH (m = 16, ef_construction = 200);

-- Hyperbolic radius B-tree (for annulus queries)
CREATE INDEX idx_nodes_hyp_radius
ON graph.nodes (hyp_radius(poincare_nav_vec))
WHERE poincare_nav_vec IS NOT NULL;
```

#### Step 5: Validate
```sql
-- Compare Poincaré KNN results against L2 KNN
-- Should show 48% neighbor divergence (matching proof-of-concept result)
WITH poinc AS (
    SELECT id, name FROM graph.nodes
    WHERE poincare_nav_vec IS NOT NULL
    ORDER BY poincare_nav_vec <~>
        (SELECT poincare_nav_vec FROM graph.nodes WHERE name = 'Python' AND type = 'technology')
    LIMIT 10
),
l2 AS (
    SELECT id, name FROM graph.nodes
    WHERE poincare_nav_vec IS NOT NULL
    ORDER BY poincare_nav_vec <->
        (SELECT poincare_nav_vec FROM graph.nodes WHERE name = 'Python' AND type = 'technology')
    LIMIT 10
)
SELECT count(*) AS shared_neighbors
FROM poinc JOIN l2 USING (id);
-- Expected: ~5 shared (48% divergence = 52% overlap ≈ 5.2 shared in top 10)
```

#### Step 6: Update LivingOS pipeline
- Modify `scripts/geometry_curate.py` to write to `poincare_nav_vec` (vector type)
  instead of `poincare_nav` (double precision[])
- Modify retrieval queries to use `<~>` operator with HNSW index
- Keep old columns for backward compatibility until verified

#### Step 7: Drop old columns (after full validation)
```sql
ALTER TABLE graph.nodes DROP COLUMN poincare_nav;
ALTER TABLE graph.nodes DROP COLUMN poincare_embedding;
ALTER TABLE graph.nodes RENAME COLUMN poincare_nav_vec TO poincare_nav;
ALTER TABLE graph.nodes RENAME COLUMN poincare_emb_vec TO poincare_embedding;
```

**Estimated migration time:** 30 minutes (mostly waiting for index builds).
**Risk:** Low — additive migration, old columns preserved until validated.
**Rollback:** Drop new columns, revert pipeline changes.

---

## Summary: Implementation Order

| Phase | What | Files | Time Est. | Milestone |
|-------|------|-------|-----------|-----------|
| 1 | Core distance + HNSW opclass | hyperbolic.h, hyperbolic.c, Makefile, .control, .sql | 4-6h | `<~>` operator + HNSW index working |
| 2 | Geometric primitives | geometric.c, extended .sql | 3-4h | midpoint, cone, Möbius, exp/log |
| 3 | IVFFlat + aggregate | aggregate.c, IVFFlat opclass | 3-4h | `hyp_centroid` + IVFFlat index |
| 4 | Benchmarks + optimization | bench/, optional Lorentz path | 2-3h | Perf numbers, recall validation |
| 5 | Advanced (curvature, layers) | Extensions to existing files | 4-6h | Variable c, research features |

**Total estimated time: 16-23 hours with AI assistance.**

Phase 1 alone produces a usable, novel system — the first indexed hyperbolic ANN
in any database. Phases 2-5 add progressively more sophisticated capabilities.

---

## Appendix: Key Formulas Reference

For quick reference during implementation. All use curvature c=1.

```
Poincaré distance:    d(x,y) = acosh(1 + 2·‖x-y‖² / ((1-‖x‖²)(1-‖y‖²)))
Hyperbolic radius:    d(0,x) = 2·atanh(‖x‖)
Conformal factor:     λ_x = 2 / (1 - ‖x‖²)
Möbius addition:      x⊕y = ((1+2⟨x,y⟩+‖y‖²)x + (1-‖x‖²)y) / (1+2⟨x,y⟩+‖x‖²‖y‖²)
Exp map at x:         exp_x(v) = x ⊕ tanh(λ_x·‖v‖/2)·v/‖v‖
Log map at x:         log_x(y) = (2/(λ_x·α))·atanh(α)·u,  u=(-x)⊕y, α=‖u‖
Midpoint:             m = (λ²_y·x + λ²_x·y) / (λ²_y + λ²_x - 2)
Cone half-aperture:   ψ(x) = arcsin(K·(1-‖x‖²)/‖x‖)
Busemann:             B_θ(x) = ln((1-‖x‖²) / ‖x-θ‖²)
Lorentz inner:        ⟨u,v⟩_L = -u₀v₀ + Σᵢuᵢvᵢ
Lorentz distance:     d_L(u,v) = acosh(-⟨u,v⟩_L)
Poincaré→Lorentz:     y₀ = (1+‖x‖²)/(1-‖x‖²),  yᵢ = 2xᵢ/(1-‖x‖²)
Lorentz→Poincaré:     xᵢ = yᵢ/(1+y₀)
Poincaré→Klein:       k = 2x/(1+‖x‖²)
Klein→Poincaré:       x = k/(1+√(1-‖k‖²))
Einstein midpoint:    k_μ = Σ(γᵢ·kᵢ)/Σ(γᵢ),  γᵢ = 1/√(1-‖kᵢ‖²)
```
