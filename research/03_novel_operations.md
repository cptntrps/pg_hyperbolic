# Novel Hyperbolic Database Operations

Research report on operations that hyperbolic (Poincare) geometry enables natively,
which are impossible or impractical with Euclidean pgvector.

**Context:** Poincare ball model B^n_c = {x in R^n : c||x||^2 < 1} with curvature -c.
All formulas use c = 1 unless noted. The conformal factor is lambda_x = 2/(1 - ||x||^2).

**Foundational formulas referenced throughout:**

```
Poincare distance:   d(x, y) = 2 arctanh(||(-x) ⊕_M y||)
Mobius addition:     x ⊕_M y = ((1 + 2<x,y> + ||y||^2)x + (1 - ||x||^2)y)
                                / (1 + 2<x,y> + ||x||^2 ||y||^2)
Conformal factor:    lambda_x = 2 / (1 - ||x||^2)
Log map at x:        log_x(y) = (2 / (lambda_x * alpha)) * arctanh(alpha) * ((-x) ⊕_M y)
                     where alpha = ||(-x) ⊕_M y||
Exp map at x:        exp_x(v) = x ⊕_M (tanh(lambda_x * ||v|| / 2) * v / ||v||)
```

---

## 1. Cone Containment Query (Entailment Cones)

### Mathematical Definition

Following Ganea et al. (ICML 2018), an entailment cone at point x in B^n
captures all points that are "more specific" than x in the learned hierarchy.

**Half-aperture angle** at point x:

```
psi(x) = arcsin(K * (1 - ||x||^2) / ||x||)
```

where K is a learned or fixed parameter (typically K = 0.1). The angle shrinks
as ||x|| grows (points near boundary have narrow cones = very specific concepts),
and widens near origin (general concepts have broad cones).

**Containment test.** Point y is in the entailment cone of x iff:

```
angle(x, y) <= psi(x)
```

where angle(x, y) is the angle at the origin between the direction to x and
the direction from x to y in the tangent space:

```
angle(x, y) = arccos(<x, log_x(y)>_x / (||x|| * ||log_x(y)||_x))
```

with the Riemannian inner product <u, v>_x = (lambda_x)^2 * <u, v>.

**Simplified test** (avoids log map, cheaper): Project y into x's tangent space
using the Mobius gyration. The test reduces to checking whether the Euclidean
angle between x and ((-x) ⊕_M y) is within psi(x):

```
IN_CONE(x, y) := arccos(<x, (-x) ⊕_M y> / (||x|| * ||(-x) ⊕_M y||)) <= psi(x)
```

This is the practical form for a SQL operator.

### Practical Use Case

"Given the entity 'Machine Learning', find all entities in its entailment cone"
returns specific subtopics: gradient descent, backpropagation, neural architecture
search — but NOT 'Computer Science' (which is more general, i.e., ML is in CS's
cone, not vice versa). This is **asymmetric** containment: the foundational
operation for IS-A / hypernymy queries on learned embeddings.

### Indexability

**Partially indexable.** A GiST index can prune candidates using a bounding-box
approximation: since the cone of x only contains points y with d(origin, y) >= d(origin, x)
(more specific = farther from origin), we can first filter by radius band
(||y|| >= ||x|| - epsilon). The angular test then runs on the filtered set.

A more sophisticated approach: build a **space-partitioning tree** (VP-tree or
ball-tree in Poincare distance). Interior nodes store bounding cones. If a node's
bounding region is entirely outside x's entailment cone, prune the subtree.

For practical PostgreSQL: a two-phase approach — GiST index on radius for band
filtering, then sequential angular check on candidates.

### SQL Function Signature

```sql
-- Operator: point @> point (cone containment)
CREATE FUNCTION hyp_cone_contains(parent poincare, child poincare, K float8 DEFAULT 0.1)
  RETURNS boolean
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Operator form
CREATE OPERATOR @> (
  LEFTARG  = poincare,
  RIGHTARG = poincare,
  FUNCTION = hyp_cone_contains
);

-- Usage: find all descendants of 'Machine Learning'
SELECT child.name
FROM entities parent, entities child
WHERE parent.name = 'Machine Learning'
  AND parent.embedding @> child.embedding;

-- With explicit K parameter
SELECT name FROM entities
WHERE hyp_cone_contains(
  (SELECT embedding FROM entities WHERE name = 'Machine Learning'),
  embedding,
  0.1
);
```

### Computational Complexity

- **Per-pair test:** O(n) where n = embedding dimension (one Mobius addition + dot product + arccos + arcsin)
- **Full scan:** O(N * n) for N entities
- **With radius pre-filter:** O(N_band * n) where N_band << N (typically 10-30% of entities)
- **Numerics:** The arcsin in psi(x) requires ||x|| > K * (1 - ||x||^2), i.e., points too close to origin have undefined (full-sphere) cones. Handle by returning TRUE for ||x|| < epsilon (the origin "contains everything").

---

## 2. Radial Band / Annulus Query

### Mathematical Definition

In the Poincare ball, the norm ||x|| encodes hierarchical depth: points near
the origin are general/abstract, points near the boundary are specific/concrete.
The hyperbolic distance from origin to x is:

```
d(0, x) = 2 arctanh(||x||)
```

An **annulus query** finds all points within a radial band [r_min, r_max] in
hyperbolic distance from the origin:

```
ANNULUS(r_min, r_max) = {x in B^n : r_min <= 2 arctanh(||x||) <= r_max}
```

Equivalently in Euclidean norm (cheaper to compute):

```
tanh(r_min / 2) <= ||x|| <= tanh(r_max / 2)
```

This converts the hyperbolic radial query to a simple Euclidean norm range check.

### Practical Use Case

"Find all entities at the same level of abstraction as 'Python'" — i.e., neither
too general ('programming') nor too specific ('Python 3.11.4 release notes').
The radius of 'Python' defines the abstraction level; the annulus query finds
peers at similar depth.

This is also the foundation for **level-of-detail rendering** in knowledge graph
UIs: zoom level maps to radius band, showing only entities at the appropriate
abstraction level.

### Indexability

**Fully indexable.** This is a 1D range query on a scalar value (||x|| or
equivalently d(0, x)). Options:

1. **Stored computed column** + B-tree index on `hyp_radius`:
   ```sql
   ALTER TABLE entities ADD COLUMN hyp_radius float8
     GENERATED ALWAYS AS (2 * atanh(poincare_norm(embedding))) STORED;
   CREATE INDEX idx_hyp_radius ON entities (hyp_radius);
   ```

2. **GiST range index** on the Poincare type itself, with the operator class
   defining a "contains radius range" strategy. This avoids the stored column
   but requires custom GiST support.

3. **Expression index** (simplest):
   ```sql
   CREATE INDEX idx_radius ON entities ((poincare_norm(embedding)));
   ```

### SQL Function Signature

```sql
-- Scalar function: hyperbolic radius (distance from origin)
CREATE FUNCTION hyp_radius(v poincare) RETURNS float8
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Annulus query operator (radius range)
CREATE FUNCTION hyp_in_annulus(v poincare, r_min float8, r_max float8)
  RETURNS boolean
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Usage: find entities at similar abstraction level as 'Python'
SELECT name FROM entities
WHERE hyp_radius(embedding)
  BETWEEN hyp_radius((SELECT embedding FROM entities WHERE name = 'Python')) - 0.2
      AND hyp_radius((SELECT embedding FROM entities WHERE name = 'Python')) + 0.2
ORDER BY hyp_dist(
  embedding,
  (SELECT embedding FROM entities WHERE name = 'Python')
);

-- Or: find all "mid-level" concepts
SELECT name FROM entities
WHERE hyp_in_annulus(embedding, 1.0, 2.0);
```

### Computational Complexity

- **Per-point test:** O(n) for norm computation, O(1) for range check
- **With B-tree index on radius:** O(log N) seek + O(k) scan for k results
- **This is the cheapest novel operation** — it reduces to a standard range query on a precomputed scalar.

---

## 3. Geodesic Midpoint (Mobius Midpoint)

### Mathematical Definition

The **Mobius midpoint** (gyromidpoint) of two points x, y in B^n is the unique
point m equidistant from both along the geodesic connecting them:

```
m = midpoint(x, y) = (lambda_y^2 * x + lambda_x^2 * y) / (lambda_y^2 + lambda_x^2 - 2 + epsilon)
```

where lambda_x = 2/(1 - ||x||^2), lambda_y = 2/(1 - ||y||^2), and epsilon is
for numerical stability (1e-15).

**Alternative via Klein model** (Einstein midpoint, numerically stabler for high dimensions):

1. Map to Klein: k = 2x / (1 + ||x||^2)
2. Compute Euclidean midpoint in Klein: k_m = (k_x + k_y) / 2
3. Map back to Poincare: m = k_m / (1 + sqrt(1 - ||k_m||^2))

The Klein approach generalizes cleanly to weighted midpoints and N-point centroids
(see operation 5).

### Practical Use Case

"What concept lies between 'Quantum Physics' and 'Computer Science'?" — the
geodesic midpoint in a well-trained embedding space should land near 'Quantum
Computing' or 'Computational Physics'. Finding nearest neighbors to the midpoint
reveals **conceptual mediators**: entities that bridge two domains.

This is fundamentally different from Euclidean midpoints because the hyperbolic
geodesic curves toward the origin (toward generality), so the mediator between
two specific concepts is appropriately more abstract than either.

### Indexability

The midpoint computation itself cannot be indexed (it is a function of two
arbitrary input points). However, the **nearest-neighbor search around the
computed midpoint** uses the standard hyperbolic distance index (HNSW or IVFFlat
with Poincare distance). So this is a two-step operation:

1. Compute midpoint: O(n) — not indexed
2. Find k-NN to midpoint: O(log N) with HNSW index

### SQL Function Signature

```sql
-- Mobius midpoint of two points
CREATE FUNCTION hyp_midpoint(a poincare, b poincare)
  RETURNS poincare
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Weighted midpoint (interpolation parameter t in [0,1])
CREATE FUNCTION hyp_midpoint(a poincare, b poincare, t float8)
  RETURNS poincare
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Usage: find conceptual mediators between two entities
SELECT name, hyp_dist(embedding, hyp_midpoint(a.embedding, b.embedding)) AS dist
FROM entities,
  (SELECT embedding FROM entities WHERE name = 'Quantum Physics') a,
  (SELECT embedding FROM entities WHERE name = 'Computer Science') b
ORDER BY dist
LIMIT 10;

-- Operator form
CREATE OPERATOR |+| (
  LEFTARG  = poincare,
  RIGHTARG = poincare,
  FUNCTION = hyp_midpoint
);
```

### Computational Complexity

- **Midpoint computation:** O(n) — two conformal factors + weighted sum + normalization
- **Nearest-neighbor around midpoint:** O(n * log N) with HNSW
- **Numerical concern:** When both points are near the boundary (||x|| -> 1), lambda values explode. Use the Klein-model path for ||x|| > 0.95.

---

## 4. Parallel Transport (Analogy Operator)

### Mathematical Definition

Parallel transport moves a tangent vector v from the tangent space T_x B^n to
T_y B^n along the geodesic from x to y, preserving its "meaning" relative to the
local geometry. In the Poincare ball:

```
PT_{x->y}(v) = (lambda_x / lambda_y) * gyr[y, -x](v)
```

where gyr[a, b] is the **gyration operator** (Thomas precession):

```
gyr[a, b](v) = -(a ⊕_M b) ⊕_M (a ⊕_M (b ⊕_M v))
```

**Simplified closed form** (from Ungar's gyrovector space theory):

```
PT_{x->y}(v) = (lambda_x / lambda_y) * A * v
```

where A is the gyration matrix. For the common case of transporting along
a geodesic, a more efficient formula exists:

```
Let u = (-x) ⊕_M y,  alpha = ||u||
PT_{x->y}(v) = v - 2 * (<log_x(y), v>_x / d(x,y)^2) * (log_x(y) + log_y(x))
```

This is the **Schild's ladder** formula adapted to hyperbolic space.

**Analogy operation:** To compute "y is to ? as b is to a":
1. Compute direction: v = log_a(b) in T_a (tangent vector from a to b)
2. Transport: w = PT_{a->y}(v) (same direction, now at y)
3. Apply: result = exp_y(w) (walk from y in the transported direction)

### Practical Use Case

The classic analogy task: "Paris is to France as Berlin is to ?" — compute the
tangent vector from 'France' to 'Paris' (the "capital-of" direction), parallel
transport it to 'Germany', and exp-map to get the answer ('Berlin' should be
the nearest entity).

In hyperbolic space this is superior to Euclidean "king - man + woman" arithmetic
because parallel transport respects curvature: the "capital-of" relationship at
different radii (abstraction levels) has different magnitudes, and transport
preserves this correctly.

### Indexability

**Not directly indexable.** This is a computation on specific input points.
However, the final exp_y(w) result is a point in B^n, and finding its nearest
neighbor IS indexable via HNSW/IVFFlat.

Pipeline: O(n) transport computation -> O(n * log N) nearest-neighbor lookup.

### SQL Function Signature

```sql
-- Parallel transport: move tangent vector v from x to y
CREATE FUNCTION hyp_parallel_transport(
  x poincare, y poincare, v poincare  -- v treated as tangent vector at x
) RETURNS poincare  -- tangent vector at y
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Full analogy: "y is to ? as b is to a"
CREATE FUNCTION hyp_analogy(
  a poincare,   -- source base
  b poincare,   -- source target
  y poincare    -- destination base
) RETURNS poincare  -- predicted destination target (point in B^n)
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Usage: "Paris is to France as ? is to Germany"
SELECT name,
  hyp_dist(embedding,
    hyp_analogy(
      (SELECT embedding FROM entities WHERE name = 'France'),
      (SELECT embedding FROM entities WHERE name = 'Paris'),
      (SELECT embedding FROM entities WHERE name = 'Germany')
    )
  ) AS dist
FROM entities
ORDER BY dist
LIMIT 5;
```

### Computational Complexity

- **Transport:** O(n) — two Mobius additions + log/exp maps (each O(n))
- **Full analogy (transport + exp + k-NN):** O(n) + O(n * log N)
- **Numerical concern:** gyration near the origin is well-conditioned; near the boundary, lambda factors amplify errors. Clamp ||x|| < 1 - epsilon (e.g., 1e-7).

---

## 5. Hyperbolic Centroid (Frechet Mean / Einstein Midpoint)

### Mathematical Definition

The **Frechet mean** of points {x_1, ..., x_N} in B^n minimizes the sum of
squared Poincare distances:

```
mu = argmin_{m in B^n} sum_i w_i * d(m, x_i)^2
```

No closed-form solution exists in the Poincare model. Two approaches:

**A. Einstein midpoint (closed-form approximation via Klein model):**

```
1. Map each x_i to Klein model:  k_i = 2 x_i / (1 + ||x_i||^2)
2. Compute Lorentz factor:       gamma_i = 1 / sqrt(1 - ||k_i||^2)
3. Weighted Einstein midpoint:   k_mu = sum(w_i * gamma_i * k_i) / sum(w_i * gamma_i)
4. Map back to Poincare:         mu = k_mu / (1 + sqrt(1 - ||k_mu||^2))
```

This is NOT the true Frechet mean but is very close for clustered points and
has O(N * n) complexity with no iteration. Used in HNN++ (Shimizu et al. 2021).

**B. Iterative Karcher flow (true Frechet mean):**

```
Repeat until convergence:
  1. v = sum_i w_i * log_{mu}(x_i)       -- weighted tangent vectors
  2. mu = exp_{mu}(eta * v)                -- step along mean direction
  3. If ||v|| < epsilon: stop
```

Converges in 5-15 iterations for well-clustered data. Each iteration is O(N * n).

### Practical Use Case

1. **IVFFlat centroids:** The inverted-file index partitions data by nearest
   centroid. In hyperbolic space, centroids MUST be computed as Frechet means,
   not Euclidean averages (which would project outside the ball or land at
   geometrically wrong positions). This is a hard requirement for any hyperbolic
   IVFFlat index.

2. **GROUP BY aggregation:** "What is the average embedding per entity type?"
   ```sql
   SELECT type, hyp_centroid(embedding) FROM entities GROUP BY type;
   ```
   This gives the geometric center of each type cluster in hyperbolic space.

3. **Cluster quality:** The Frechet variance (sum of squared distances to centroid)
   measures cluster tightness in the correct geometry.

### Indexability

The centroid itself is a **computation, not a query** — it is consumed by index
construction (IVFFlat) and aggregation. The Einstein midpoint's O(N * n)
complexity makes it suitable for index build time. The iterative Karcher flow
is too slow for index construction but fine for small aggregation groups.

### SQL Function Signature

```sql
-- Aggregate function: hyperbolic centroid
CREATE AGGREGATE hyp_centroid(poincare) (
  SFUNC     = hyp_centroid_accum,      -- accumulates Klein-projected weighted sums
  STYPE     = internal,
  FINALFUNC = hyp_centroid_final,       -- divides by sum of gammas, projects back
  PARALLEL  = SAFE
);

-- Weighted variant
CREATE AGGREGATE hyp_centroid(poincare, float8) (
  SFUNC     = hyp_centroid_weighted_accum,
  STYPE     = internal,
  FINALFUNC = hyp_centroid_final,
  PARALLEL  = SAFE
);

-- Non-aggregate function form (for arrays)
CREATE FUNCTION hyp_centroid(points poincare[]) RETURNS poincare
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Usage: type-level centroids
SELECT type, hyp_centroid(embedding) AS center,
       avg(hyp_dist(embedding, hyp_centroid(embedding) OVER (PARTITION BY type))) AS spread
FROM entities
GROUP BY type;

-- Usage: IVFFlat build (internal, called by index AM)
-- The index access method calls hyp_centroid on each partition during build.
```

### Computational Complexity

- **Einstein midpoint:** O(N * n) — single pass, no iteration. Recommended default.
- **Karcher flow:** O(I * N * n) where I = 5-15 iterations. For precision-sensitive cases.
- **As aggregate:** State is O(n) (running weighted sum in Klein space + scalar gamma sum). Very memory-efficient.
- **Parallel safety:** Einstein midpoint is trivially parallelizable — partial sums from parallel workers can be merged by adding Klein-projected weighted sums and gamma sums.

---

## 6. Hyperbolic Interpolation (Geodesic Walk)

### Mathematical Definition

Given two points x, y in B^n, compute the point at fraction t in [0, 1] along
the geodesic from x to y:

```
gamma(t) = x ⊕_M (tanh(t * arctanh(||u||)) * u / ||u||)
where u = (-x) ⊕_M y
```

Equivalently using log/exp maps:

```
gamma(t) = exp_x(t * log_x(y))
```

For a sequence of N evenly-spaced points along the geodesic:

```
gamma_k = gamma(k / (N-1))  for k = 0, 1, ..., N-1
```

### Practical Use Case

**Concept gradients:** "Show me the spectrum of concepts between 'Physics' and
'Philosophy'." Interpolating along the geodesic and finding nearest neighbors at
each step reveals intermediate concepts: 'Philosophy of Science', 'Epistemology
of Physics', 'Foundations of Quantum Mechanics', etc.

This is fundamentally different from Euclidean interpolation because the
hyperbolic geodesic curves through regions of appropriate abstraction. Two
specific entities interpolate through more general intermediate territory
(the geodesic passes closer to the origin), matching how conceptual bridges
between specific topics are typically more abstract.

### Indexability

**Not indexable as a single operation.** Each interpolated point is computed on
the fly, and its nearest neighbors are found via the standard HNSW/IVFFlat index.
For N interpolation steps, this is N independent k-NN queries.

**Batch optimization:** Compute all N points first, then batch-query the index.
If the index supports batch k-NN (as HNSW does), this reduces overhead from
lock acquisition and graph entry-point computation.

### SQL Function Signature

```sql
-- Single interpolation point
CREATE FUNCTION hyp_lerp(a poincare, b poincare, t float8)
  RETURNS poincare
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Generate N points along geodesic (set-returning)
CREATE FUNCTION hyp_geodesic_walk(a poincare, b poincare, steps int)
  RETURNS SETOF poincare
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE ROWS 10;

-- Usage: concept gradient between two entities
SELECT step.t, neighbor.name, hyp_dist(step.point, neighbor.embedding) AS dist
FROM (
  SELECT ordinality AS t, point
  FROM hyp_geodesic_walk(
    (SELECT embedding FROM entities WHERE name = 'Physics'),
    (SELECT embedding FROM entities WHERE name = 'Philosophy'),
    10
  ) WITH ORDINALITY AS point
) step
CROSS JOIN LATERAL (
  SELECT name, embedding
  FROM entities
  ORDER BY hyp_dist(embedding, step.point)
  LIMIT 3
) neighbor;
```

### Computational Complexity

- **Single interpolation point:** O(n) — one Mobius addition + tanh + scalar ops
- **N-step walk:** O(N * n) for point generation
- **With k-NN at each step:** O(N * n * log M) where M = dataset size (using HNSW)
- **Memory:** O(n) per step — can be streamed via SRF (set-returning function)

---

## 7. Busemann Function Queries

### Mathematical Definition

The **Busemann function** measures progress toward a point at infinity (an ideal
point on the boundary of the Poincare ball). Given a unit direction
theta in S^{n-1} (a point on the boundary ||theta|| = 1), the Busemann function
at point x is:

```
B_theta(x) = ln((1 - ||x||^2) / (||x - theta||^2))
     = ln(1 - ||x||^2) - ln(||x||^2 - 2<x, theta> + 1)
```

**Level sets** of B_theta are **horospheres** — hyperbolic analogs of planes
perpendicular to the direction theta. All points on a horosphere are "equally
far along" in direction theta.

**Directional query:** "Find entities in direction theta with Busemann value in [b_min, b_max]":

```
{x in B^n : b_min <= B_theta(x) <= b_max}
```

Higher B_theta(x) means x is "further along" in direction theta. Negative values
mean x is "behind" the origin relative to theta.

### Practical Use Case

Busemann queries answer: "Find all entities in the direction of [concept] at
infinity, regardless of how specific they are." Unlike cone containment (which
requires a specific apex point), Busemann queries use a **direction** — an
idealized limit concept that may not correspond to any actual entity.

Example: If the direction theta points toward where 'pure mathematics' concepts
cluster at the boundary, `B_theta(x) > 0` finds all entities with mathematical
character — from general ('Mathematics') to specific ('Zorn's Lemma') — filtering
by direction without filtering by abstraction level.

This is also the natural query for **horosphere-based clustering**: partition
entities by which horosphere they lie on relative to a set of ideal directions.
Ganea's "Hyperbolic Busemann Learning" (NeurIPS 2021) uses exactly this for
prototype-based classification.

### Indexability

**Partially indexable.** For a fixed set of "important" directions {theta_1, ..., theta_K},
precompute B_{theta_k}(x) as stored columns and index with B-trees. This gives
O(log N) lookup for queries along those K directions.

For arbitrary directions: the Busemann function involves only dot products and
norms, so it can be evaluated in O(n) per point. No general-purpose index
structure exists, but a **space-partitioning index** (KD-tree variant in
Poincare space) could provide moderate pruning.

### SQL Function Signature

```sql
-- Busemann function value for point x toward ideal direction theta
CREATE FUNCTION hyp_busemann(x poincare, theta poincare)
  RETURNS float8
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
-- Note: theta should have ||theta|| = 1 (boundary point).
-- If ||theta|| < 1, it is normalized internally.

-- Query: entities in a direction band
SELECT name, hyp_busemann(embedding, '{0.98, 0.0, ...}'::poincare) AS progress
FROM entities
WHERE hyp_busemann(embedding, '{0.98, 0.0, ...}'::poincare) BETWEEN 0.5 AND 2.0
ORDER BY progress DESC;

-- Usage: find entities most aligned with a direction
SELECT name FROM entities
ORDER BY hyp_busemann(embedding, normalize_direction(
  (SELECT embedding FROM entities WHERE name = 'Pure Mathematics')
)) DESC
LIMIT 20;

-- Helper: project a point to the boundary (get its "direction at infinity")
CREATE FUNCTION hyp_ideal_direction(x poincare)
  RETURNS poincare  -- unit vector: x / ||x||
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
```

### Computational Complexity

- **Per-point evaluation:** O(n) — one dot product + two norm computations + ln
- **Full scan:** O(N * n)
- **With precomputed columns for K directions:** O(log N) per direction (B-tree)
- **Numerical concern:** As ||x|| -> 1, the (1 - ||x||^2) term approaches 0 and B_theta -> -infinity unless x is aligned with theta. Use log1p for stability: `log1p(-||x||^2)`.

---

## 8. Distance to Geodesic

### Mathematical Definition

Given a geodesic G defined by two points a, b in B^n, the distance from point x
to G is:

```
d(x, G) = min_{t in [0,1]} d(x, gamma(t))
```

where gamma(t) is the geodesic from a to b (see operation 6).

**Closed-form approach** using the hyperbolic law of cosines. In the Poincare ball,
the geodesic through a and b lies on a circle (or diameter) orthogonal to the
boundary. The perpendicular distance from x to this geodesic is:

```
d(x, G) = arcsinh(|sinh(d(a,x)) * sin(angle_a)|)
```

where angle_a is the angle at vertex a in the hyperbolic triangle (a, b, x):

```
cos(angle_a) = (cosh(d(a,b)) * cosh(d(a,x)) - cosh(d(b,x)))
               / (sinh(d(a,b)) * sinh(d(a,x)))
```

This is derived from the hyperbolic sine rule. All cosh/sinh terms can be computed
from Poincare distances.

**Alternative (tangent space projection):**

```
v_ab = log_a(b)          -- tangent direction of geodesic at a
v_ax = log_a(x)          -- tangent direction to x at a
proj = <v_ax, v_ab>_a / <v_ab, v_ab>_a  -- scalar projection
perp = v_ax - proj * v_ab               -- perpendicular component
d(x, G) ≈ (2 / lambda_a) * ||perp||     -- approximate distance (exact for small distances)
```

The tangent-space method is cheaper but approximate. The hyperbolic law of cosines
method is exact.

### Practical Use Case

"Find all concepts along the path between 'Biology' and 'Computer Science'" — this
query returns entities close to the geodesic connecting the two, revealing the
conceptual bridge: 'Bioinformatics', 'Computational Biology', 'Genomics', 'Systems
Biology', etc.

Unlike Euclidean distance-to-line (which would find a linear interpolation), the
hyperbolic geodesic curves through the appropriate abstraction level, and distance-
to-geodesic respects the exponential expansion of hyperbolic space: entities
"slightly off the path" at high specificity (near boundary) are actually very
far in hyperbolic terms, giving correct specificity-aware filtering.

### Indexability

**Not directly indexable.** The geodesic is defined by two arbitrary query points,
so no precomputation is possible. However, two pruning strategies exist:

1. **Radius band pruning:** The geodesic between a and b has a minimum and maximum
   radius. Points far outside this band can be excluded via a B-tree index on
   hyp_radius (see operation 2).

2. **Triangle inequality pruning:** d(x, G) >= max(0, d(x, a) - d(a, b)). If
   the HNSW index gives d(x, a) cheaply, candidates far from both a and b can
   be pruned.

For practical use: retrieve top-K candidates near both a and b (via HNSW), then
compute exact distance-to-geodesic on the union of candidates.

### SQL Function Signature

```sql
-- Distance from point to geodesic defined by two points
CREATE FUNCTION hyp_dist_to_geodesic(x poincare, geo_a poincare, geo_b poincare)
  RETURNS float8
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Find entities along the path between two points
SELECT name,
  hyp_dist_to_geodesic(embedding, a.embedding, b.embedding) AS path_dist
FROM entities,
  (SELECT embedding FROM entities WHERE name = 'Biology') a,
  (SELECT embedding FROM entities WHERE name = 'Computer Science') b
WHERE hyp_dist_to_geodesic(embedding, a.embedding, b.embedding) < 0.5
ORDER BY path_dist;

-- "Geodesic corridor" operator (boolean: within threshold of geodesic)
CREATE FUNCTION hyp_near_geodesic(
  x poincare, geo_a poincare, geo_b poincare, threshold float8
) RETURNS boolean
  LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Combine with LATERAL for efficient execution
SELECT name, path_dist
FROM (SELECT embedding FROM entities WHERE name = 'Biology') a,
     (SELECT embedding FROM entities WHERE name = 'Computer Science') b,
LATERAL (
  SELECT name, hyp_dist_to_geodesic(e.embedding, a.embedding, b.embedding) AS path_dist
  FROM entities e
  WHERE hyp_in_annulus(e.embedding,
    LEAST(hyp_radius(a.embedding), hyp_radius(b.embedding)) - 0.5,
    GREATEST(hyp_radius(a.embedding), hyp_radius(b.embedding)) + 0.5
  )
  ORDER BY path_dist
  LIMIT 20
) results;
```

### Computational Complexity

- **Per-point (law of cosines):** O(n) — three Poincare distance computations + hyperbolic trig
- **Per-point (tangent projection):** O(n) — two log maps + dot products (cheaper constant)
- **Full scan:** O(N * n)
- **With radius + triangle pruning:** O(N_pruned * n), typically N_pruned = 5-20% of N
- **Numerical concern:** When d(a, b) is very small (geodesic degenerates to a point), the formula is unstable. Fall back to d(x, a) when d(a, b) < epsilon.

---

## Summary: Operation Comparison Matrix

| # | Operation | Complexity (per point) | Indexable? | Novel vs Euclidean |
|---|-----------|----------------------|------------|-------------------|
| 1 | Cone containment | O(n) | Partial (radius pre-filter) | **Impossible** in Euclidean — no intrinsic asymmetric containment |
| 2 | Annulus query | O(n) compute, O(1) check | **Full** (B-tree on radius) | Trivial in Euclidean but semantically meaningless (no abstraction hierarchy) |
| 3 | Geodesic midpoint | O(n) | Via k-NN on result | Euclidean midpoint exists but lacks curvature-aware abstraction |
| 4 | Parallel transport | O(n) | Via k-NN on result | **Impossible** in Euclidean — flat space has trivial (identity) transport |
| 5 | Frechet mean | O(N*n) per group | N/A (aggregate) | Euclidean mean exists but distorts hyperbolic structure |
| 6 | Geodesic walk | O(n) per step | Via k-NN per step | Euclidean lerp exists but path is wrong (straight line vs curved) |
| 7 | Busemann function | O(n) | Partial (precomputed dirs) | **Impossible** in Euclidean — no boundary / ideal points |
| 8 | Dist to geodesic | O(n) | Partial (radius + triangle) | Euclidean dist-to-line exists but misses curvature effects |

### Priority Ranking for Implementation

1. **Frechet mean / Einstein midpoint** — REQUIRED for IVFFlat index and GROUP BY. No workaround.
2. **Annulus query (hyp_radius)** — trivial to implement, huge practical value, fully indexable.
3. **Geodesic midpoint** — simple formula, immediately useful for "concept bridging" queries.
4. **Cone containment** — the signature operation that justifies hyperbolic embeddings for hierarchies.
5. **Geodesic walk** — natural extension of midpoint, enables concept gradient UIs.
6. **Distance to geodesic** — powerful for path-based exploration, moderate implementation effort.
7. **Parallel transport / analogy** — mathematically rich but niche use case in KG context.
8. **Busemann function** — most exotic, valuable for direction-based clustering and prototype learning.

---

## References

- [Ganea et al. — Hyperbolic Entailment Cones (ICML 2018)](http://proceedings.mlr.press/v80/ganea18a/ganea18a.pdf)
- [Nickel & Kiela — Poincare Embeddings (NeurIPS 2017)](https://arxiv.org/pdf/1705.08039)
- [Shimizu et al. — Hyperbolic Neural Networks++ (ICLR 2021)](https://arxiv.org/pdf/2006.08210)
- [Lou et al. — Differentiating through the Frechet Mean (ICML 2020)](http://proceedings.mlr.press/v119/lou20a/lou20a.pdf)
- [Atigh et al. — Hyperbolic Busemann Learning (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/01259a0cb2431834302abe2df60a1327-Paper.pdf)
- [Wilson — Gradient Descent in Hyperbolic Space (2018)](https://arxiv.org/pdf/1805.08207)
- [Busemann Function — Wikipedia](https://en.wikipedia.org/wiki/Busemann_function)
- [PostgreSQL GiST Index Documentation](https://www.postgresql.org/docs/current/gist.html)
- [Shadow Cones (ICLR 2024)](https://arxiv.org/pdf/2305.15215)
