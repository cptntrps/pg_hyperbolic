# pgvector Internals Research (v0.8.1)

Target: PostgreSQL 16.12, pgvector 0.8.1

Source: https://github.com/pgvector/pgvector/tree/v0.8.1

---

## 1. Distance Function Architecture

### 1.1 Core Distance Functions (src/vector.c)

All distance functions follow the same pattern: a static computation function for the math, and a PG_FUNCTION_INFO_V1-registered wrapper that extracts Vector arguments and returns float8.

```c
// Static computation function (auto-vectorized via compiler flags)
VECTOR_TARGET_CLONES static float
VectorL2SquaredDistance(int dim, float *ax, float *bx)
{
    float distance = 0.0;
    for (int i = 0; i < dim; i++) {
        float diff = ax[i] - bx[i];
        distance += diff * diff;
    }
    return distance;
}

// PG-registered wrapper
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_l2_squared_distance);
Datum
vector_l2_squared_distance(PG_FUNCTION_ARGS)
{
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    PG_RETURN_FLOAT8((double) VectorL2SquaredDistance(a->dim, a->x, b->x));
}
```

**Key distance functions defined in vector.c:**

| C Function | Purpose | Used By Index |
|---|---|---|
| `l2_distance` | sqrt(sum((a-b)^2)) | User-facing `<->` |
| `vector_l2_squared_distance` | sum((a-b)^2) — no sqrt | HNSW/IVF FUNCTION 1 for l2_ops |
| `inner_product` | sum(a*b) | User-facing |
| `vector_negative_inner_product` | -sum(a*b) | HNSW/IVF FUNCTION 1 for ip_ops/cosine_ops |
| `cosine_distance` | 1 - cos_sim | User-facing `<=>` |
| `vector_spherical_distance` | acos(dot)/pi — angular distance | IVF FUNCTION 3 (k-means) for ip/cosine |
| `l1_distance` | sum(|a-b|) | HNSW FUNCTION 1 for l1_ops |
| `vector_norm` | sqrt(sum(a^2)) | FUNCTION 2 (norm check) for cosine_ops |

**Critical insight:** The index uses **different functions** from what the user-facing operator calls. The `<->` operator calls `l2_distance` (with sqrt), but the HNSW/IVF index internally uses `vector_l2_squared_distance` (without sqrt). This is valid because sqrt is monotonic, so ordering is preserved.

Similarly, cosine_ops internally stores normalized vectors and uses `vector_negative_inner_product` as the distance function (FUNCTION 1), with `vector_norm` as the normalization function (FUNCTION 2).

### 1.2 Operator Registration (sql/vector.sql)

Operators are defined as SQL-level mappings from operator symbols to C functions:

```sql
CREATE OPERATOR <-> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = vector_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE OPERATOR <+> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = l1_distance,
    COMMUTATOR = '<+>'
);
```

### 1.3 Operator Class Registration

Each distance metric gets an operator class per index type. The operator class binds the operator, the index-internal distance function, and optional support functions:

```sql
-- IVFFlat has 5 support functions:
--   FUNCTION 1: distance (for scan)
--   FUNCTION 2: norm (for normalization check)
--   FUNCTION 3: k-means distance (must satisfy triangle inequality)
--   FUNCTION 4: k-means norm
--   FUNCTION 5: type info

CREATE OPERATOR CLASS vector_l2_ops
    DEFAULT FOR TYPE vector USING ivfflat AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_squared_distance(vector, vector),
    FUNCTION 3 l2_distance(vector, vector);  -- k-means uses L2 (not squared)

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING ivfflat AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector),
    FUNCTION 3 vector_spherical_distance(vector, vector),  -- angular for k-means
    FUNCTION 4 vector_norm(vector);  -- normalize for spherical k-means

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING ivfflat AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector),
    FUNCTION 2 vector_norm(vector),  -- normalize inputs
    FUNCTION 3 vector_spherical_distance(vector, vector),
    FUNCTION 4 vector_norm(vector);

-- HNSW has 3 support functions:
--   FUNCTION 1: distance
--   FUNCTION 2: norm (for normalization)
--   FUNCTION 3: type info

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_squared_distance(vector, vector);

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector);

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector),
    FUNCTION 2 vector_norm(vector);

CREATE OPERATOR CLASS vector_l1_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <+> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 l1_distance(vector, vector);
```

### 1.4 Support Function Constants (src/hnsw.h, src/ivfflat.h)

```c
// HNSW support function numbers
#define HNSW_DISTANCE_PROC   1
#define HNSW_NORM_PROC       2
#define HNSW_TYPE_INFO_PROC  3

// IVFFlat support function numbers
#define IVFFLAT_DISTANCE_PROC        1
#define IVFFLAT_NORM_PROC            2
#define IVFFLAT_KMEANS_DISTANCE_PROC 3
#define IVFFLAT_KMEANS_NORM_PROC     4
#define IVFFLAT_TYPE_INFO_PROC       5
```

---

## 2. HNSW Implementation

### 2.1 Index Handler (src/hnsw.c)

The HNSW access method is registered via `hnswhandler`, which returns an `IndexAmRoutine` struct:

```c
amroutine->amsupport = 3;           // 3 support functions
amroutine->amcanorderbyop = true;   // supports ORDER BY operator
amroutine->ambuild = hnswbuild;
amroutine->aminsert = hnswinsert;
amroutine->ambeginscan = hnswbeginscan;
amroutine->amrescan = hnswrescan;
amroutine->amgettuple = hnswgettuple;
```

### 2.2 Distance Computation in HNSW

All distance computation flows through a single bottleneck in `hnswutils.c`:

```c
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport *support)
{
    return DatumGetFloat8(FunctionCall2Coll(
        support->procinfo, support->collation, a, b));
}
```

Where `support->procinfo` is loaded from `HNSW_DISTANCE_PROC` (support function 1) in the operator class:

```c
void HnswInitSupport(HnswSupport *support, Relation index)
{
    support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
    support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
    support->collation = index->rd_indcollation[0];
}
```

**This is completely generic.** HNSW does not hardcode any distance formula. It always calls `support->procinfo` via the PostgreSQL function call interface. The distance function is determined entirely by the operator class.

### 2.3 SearchLayer (Algorithm 2) — Metric Assumptions

The core search (src/hnswutils.c, `HnswSearchLayer`) implements the standard HNSW greedy search:

```c
while (!pairingheap_is_empty(C)) {
    HnswSearchCandidate *c = ... pairingheap_remove_first(C);
    HnswSearchCandidate *f = ... pairingheap_first(W);

    if (c->distance > f->distance)
        break;  // <-- ASSUMES: distance ordering is meaningful

    // For each unvisited neighbor:
    //   compute eDistance
    //   if (eDistance < f->distance || alwaysAdd)
    //     add to candidates
}
```

**Metric assumptions in HNSW search:**
1. **Monotonicity of comparison:** The algorithm only compares distances. It never adds, subtracts, or averages them. It uses `<` and `>` comparisons only.
2. **No triangle inequality required for search correctness.** The search is a greedy beam search — triangle inequality improves recall but isn't required for the algorithm to terminate or produce results.
3. **Symmetry assumed:** The distance function is called as `d(query, element)` only, never `d(element, query)`. The COMMUTATOR on the operator implies symmetry.

### 2.4 SelectNeighbors (Algorithm 4) — Pruning Heuristic

The neighbor pruning (Malkov/Yashunin heuristic) in `SelectNeighbors`:

```c
static bool
CheckElementCloser(char *base, HnswCandidate *e, List *r, HnswSupport *support)
{
    Datum eValue = HnswGetValue(base, eElement);
    foreach(lc2, r) {
        HnswCandidate *ri = lfirst(lc2);
        Datum riValue = HnswGetValue(base, riElement);
        float distance = HnswGetDistance(eValue, riValue, support);

        if (distance <= e->distance)
            return false;  // <-- USES triangle-inequality-like reasoning
    }
    return true;
}
```

**This is where triangle inequality matters most.** The pruning heuristic says: "keep candidate `e` only if no already-selected neighbor `ri` is closer to `e` than `e` is to the query." This reasoning is geometrically motivated by triangle inequality: if `d(e,ri) <= d(e,q)`, then going through `ri` is at least as good as going through `e`.

**For Poincare distance:** This heuristic will still work because Poincare distance IS a true metric — it satisfies triangle inequality, symmetry, and positivity. The heuristic may be less effective in high-curvature regions where the triangle inequality is "loose," but it will not produce incorrect results.

### 2.5 HnswUpdateConnection — Neighbor Update

When inserting, bidirectional connections are managed:

```c
void HnswUpdateConnection(char *base, HnswNeighborArray *neighbors,
    HnswElement newElement, float distance, int lm, ...)
{
    newHc.distance = distance;  // stores float distance

    if (neighbors->length < lm) {
        neighbors->items[neighbors->length++] = newHc;
    } else {
        // Calls SelectNeighbors to prune
    }
}
```

**Distance is stored as `float` (32-bit) in `HnswCandidate.distance`.** The search uses `double` for `HnswSearchCandidate.distance`. No special handling of distance ranges.

### 2.6 Type Info System

HNSW uses a `HnswTypeInfo` struct to handle different vector types:

```c
typedef struct HnswTypeInfo {
    int         maxDimensions;
    Datum       (*normalize)(PG_FUNCTION_ARGS);
    void        (*checkValue)(Pointer v);
} HnswTypeInfo;
```

This is retrieved via `HNSW_TYPE_INFO_PROC` (support function 3), falling back to a default for the `vector` type. The `halfvec`, `bit`, and `sparsevec` types provide their own via support functions like `hnsw_halfvec_support`.

---

## 3. IVFFlat Implementation

### 3.1 Index Handler (src/ivfflat.c)

```c
amroutine->amsupport = 5;  // 5 support functions (vs HNSW's 3)
```

### 3.2 K-Means Clustering (src/ivfkmeans.c)

IVFFlat uses Elkan's accelerated k-means, which **explicitly requires triangle inequality**:

```c
/*
 * Use Elkan for performance. This requires distance function to
 * satisfy triangle inequality.
 *
 * We use L2 distance for L2 (not L2 squared like index scan)
 * and angular distance for inner product and cosine distance
 */
static void
ElkanKmeans(Relation index, VectorArray samples, VectorArray centers, ...)
{
    // Uses IVFFLAT_KMEANS_DISTANCE_PROC (support function 3)
    procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
    ...
}
```

**Critical detail:** IVFFlat uses DIFFERENT distance functions for k-means vs scan:
- **Scan (FUNCTION 1):** `vector_l2_squared_distance` — fast, no sqrt, preserves ordering
- **K-means (FUNCTION 3):** `l2_distance` — with sqrt, satisfies triangle inequality

For inner product / cosine, k-means uses `vector_spherical_distance` (angular distance = acos(dot)/pi), which is a true metric.

### 3.3 Centroid Computation

Centroids are computed as arithmetic means (standard k-means):

```c
// ComputeNewCenters in ivfkmeans.c
for (int k = 0; k < dimensions; k++)
    x[k] /= centerCounts[j];  // simple averaging
```

**For Poincare space:** This is wrong. Arithmetic mean is not the Frechet/Karcher mean in hyperbolic space. IVFFlat centroids would need to be computed differently (e.g., Einstein midpoint or iterative Frechet mean).

The centroid computation is abstracted through `IvfflatTypeInfo`:

```c
typedef struct IvfflatTypeInfo {
    int         maxDimensions;
    Datum       (*normalize)(PG_FUNCTION_ARGS);
    Size        (*itemSize)(int dimensions);
    void        (*updateCenter)(Pointer v, int dimensions, float *x);
    void        (*sumCenter)(Pointer v, float *x);
} IvfflatTypeInfo;
```

`sumCenter` accumulates element-wise sums, `updateCenter` writes the final centroid from accumulated `float *x` values. The division happens in `ComputeNewCenters` before calling `updateCenter`. A custom type would need to override both to implement hyperbolic averaging.

### 3.4 Scan Distance (src/ivfscan.c)

```c
// In ivfflatbeginscan:
so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);

// During scan, distance is computed via function pointer:
so->distfunc = FunctionCall2Coll;  // or ZeroDistance for NULL
distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, ...));
```

Like HNSW, the scan-time distance is fully generic — it calls whatever is registered as FUNCTION 1.

---

## 4. Adding a New Distance Type — Minimal Change Set

### 4.1 What's Needed

To add a Poincare distance metric, you need:

1. **C distance function** — the Poincare ball distance computation
2. **SQL operator** — a new operator symbol (e.g., `<~>` is taken by hamming; could use a new one)
3. **Operator classes** — one per index type (HNSW, IVFFlat)
4. **Optional: normalization function** — to project onto the ball if needed
5. **Optional: k-means distance** — for IVFFlat, a triangle-inequality-satisfying variant

### 4.2 File Modification Map (if forking pgvector)

| File | Change |
|---|---|
| `src/vector.c` | Add `poincare_distance()` C function + PG wrapper |
| `sql/vector.sql` | Add operator, operator classes for hnsw + ivfflat |
| `src/vector.h` | (no change needed - Vector struct reused) |
| `src/hnsw.h` | (no change - distance is generic) |
| `src/ivfflat.h` | (no change - distance is generic) |
| `src/hnswutils.c` | (no change - distance via FunctionCall2Coll) |
| `src/ivfkmeans.c` | (no change IF Poincare distance satisfies triangle inequality; it does) |

**The minimal set is literally:**
1. One new C function in `vector.c` (or a new file)
2. SQL additions in `vector.sql` (operator + operator classes)

### 4.3 Registration Mechanism

The registration is entirely SQL-driven via operator classes. The C code only needs to provide PG_FUNCTION_INFO_V1-registered functions. The index code looks up the function via `index_getprocinfo(index, 1, DISTANCE_PROC)` — it never checks what the function actually is.

---

## 5. Extension vs Fork

### 5.1 Can This Be a Separate Extension?

**Yes, partially.** A separate extension can:
- Define new SQL functions (`poincare_distance`)
- Define new operators
- Define new operator classes that reference the existing `hnsw` and `ivfflat` access methods

This works because:
1. The access methods (`hnsw`, `ivfflat`) are registered in the PostgreSQL catalog and accessible to any extension
2. Operator classes are catalog entries — any extension can create them
3. The HNSW/IVF code resolves distance functions at runtime via `index_getprocinfo`

**Example of what a standalone extension's SQL would look like:**

```sql
-- poincare.sql
CREATE FUNCTION poincare_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OPERATOR <@> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = poincare_distance,
    COMMUTATOR = '<@>'
);

-- This references pgvector's hnsw access method
CREATE OPERATOR CLASS vector_poincare_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <@> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector);

-- Same for ivfflat
CREATE OPERATOR CLASS vector_poincare_ops
    FOR TYPE vector USING ivfflat AS
    OPERATOR 1 <@> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector),
    FUNCTION 3 poincare_distance(vector, vector);  -- Poincare IS a metric
```

### 5.2 Limitations of the Extension Approach

1. **Dependency management:** The extension must declare pgvector as a dependency (via `requires = 'vector'` in the .control file). PostgreSQL handles extension load order.

2. **IVFFlat centroid computation:** The `sumCenter`/`updateCenter` functions in `IvfflatTypeInfo` assume Euclidean averaging. To provide hyperbolic centroids, you'd need to register a custom `TYPE_INFO_PROC` (FUNCTION 5 for IVFFlat). This IS extensible — `IvfflatGetTypeInfo` checks for a custom proc first:

    ```c
    const IvfflatTypeInfo *
    IvfflatGetTypeInfo(Relation index)
    {
        FmgrInfo *procinfo = IvfflatOptionalProcInfo(index, IVFFLAT_TYPE_INFO_PROC);
        if (procinfo == NULL) {
            // return default (Euclidean)
        } else {
            return (const IvfflatTypeInfo *) DatumGetPointer(
                FunctionCall0Coll(procinfo, InvalidOid));
        }
    }
    ```

    But providing custom `sumCenter`/`updateCenter` requires C function pointers that operate on the raw float arrays, which means the extension needs to implement the Frechet mean logic in C.

3. **The `vector` type is reused.** Poincare vectors are just float arrays — the same `Vector` struct works. No need for a new type. However, there's no constraint enforcement at the type level to ensure vectors are inside the unit ball (norm < 1). This would need to be handled at insert time via a check constraint or a custom normalization function.

4. **HNSW type info:** `HnswTypeInfo` includes `checkValue` which could validate that vectors are within the Poincare ball:

    ```c
    typedef struct HnswTypeInfo {
        int         maxDimensions;
        Datum       (*normalize)(PG_FUNCTION_ARGS);
        void        (*checkValue)(Pointer v);
    } HnswTypeInfo;
    ```

    A custom `HNSW_TYPE_INFO_PROC` (FUNCTION 3) can provide this.

### 5.3 Recommendation

**Start as a standalone extension.** The architecture supports it cleanly:
- HNSW works out of the box with any distance function (just register operator class)
- IVFFlat works for scan, but centroid computation needs custom `TYPE_INFO_PROC` for correct hyperbolic means
- No need to fork pgvector — all extension points exist

**Fall back to fork only if:**
- Performance of `FunctionCall2Coll` overhead is unacceptable (unlikely — it's the same overhead pgvector itself uses)
- You need to modify the HNSW pruning heuristic for better recall in hyperbolic space
- You need Poincare-specific optimizations in the neighbor selection logic

---

## 6. Data Structures Reference

### 6.1 Vector Type (src/vector.h)

```c
typedef struct Vector {
    int32   vl_len_;    // varlena header
    int16   dim;        // number of dimensions
    int16   unused;     // reserved, always zero
    float   x[FLEXIBLE_ARRAY_MEMBER];  // float32 components
} Vector;

#define VECTOR_MAX_DIM 16000
#define VECTOR_SIZE(_dim) (offsetof(Vector, x) + sizeof(float)*(_dim))
```

### 6.2 Build System (Makefile)

Standard PGXS extension build:

```makefile
EXTENSION = vector
MODULE_big = vector
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o \
       src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o \
       src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o \
       src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o \
       src/ivfvacuum.o src/sparsevec.o src/vector.o
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math \
             -fno-signed-zeros -fno-trapping-math
```

A separate extension would have its own Makefile with just its .o files, linking against pgvector's headers (exported: `vector.h`, `halfvec.h`, `sparsevec.h`).

---

## 7. Summary of Key Findings

1. **Distance functions are fully pluggable.** Both HNSW and IVFFlat resolve distance functions at runtime via PostgreSQL's operator class mechanism. No hardcoded distance logic exists in the index code.

2. **HNSW has zero metric assumptions in its search algorithm.** It only compares distances. The pruning heuristic benefits from triangle inequality but doesn't require it. Poincare distance (a true metric) will work correctly.

3. **IVFFlat has two concerns:**
   - K-means uses Elkan's algorithm which requires triangle inequality (Poincare distance satisfies this)
   - Centroid computation assumes Euclidean averaging (must be overridden via `TYPE_INFO_PROC`)

4. **A standalone extension is viable** for HNSW immediately. For IVFFlat, it requires implementing custom centroid computation via the TYPE_INFO_PROC support function.

5. **The Vector struct can be reused.** Poincare ball vectors are just float arrays. Ball containment (norm < 1) would need validation via `checkValue` in the type info.

6. **Minimal implementation path:** One C file with the Poincare distance function + one SQL file with operator and operator class definitions = working HNSW index with Poincare distance.
