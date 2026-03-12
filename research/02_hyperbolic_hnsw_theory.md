# Hyperbolic HNSW: Mathematical and Algorithmic Feasibility

**Date:** 2026-03-12
**Focus:** Can HNSW work natively in hyperbolic (Poincare) space? What modifications are needed?

---

## 1. HNSW in Non-Euclidean Spaces: Published Work

### 1.1 Graph-Based NNS in Hyperbolic Spaces (Prokhorenkova et al., ICLR 2022)

The most directly relevant paper is "Graph-based Nearest Neighbor Search in Hyperbolic Spaces" (Prokhorenkova, Shekhovtsov, ICLR 2022), which provides rigorous complexity analysis of graph-based NNS (including HNSW) operating directly in hyperbolic space.

**Key results:**
- Graph-based NNS applied directly to hyperbolic space **outperforms** reduction-based approaches (reducing to Euclidean NNS then solving there).
- In the **dense regime** (d << log n), graph-based NNS in hyperbolic space achieves **lower time complexity** than the Euclidean case.
- The exponential volume growth of hyperbolic balls is an **advantage**, not a problem: all elements become closer to each other relative to the ball radius, reducing the number of greedy hops to O(log n) even without long-range edges.
- HNSW significantly outperforms all baselines in empirical evaluation.
- **Conclusion:** "Graph-based NNS can be considered a default approach for similarity search in hyperbolic spaces."

**Citation:** Prokhorenkova, L., Shekhovtsov, A. "Graph-based Nearest Neighbor Search in Hyperbolic Spaces." ICLR 2022.
URL: https://openreview.net/forum?id=USIgIY6TNDe

### 1.2 NNS for Hyperbolic Embeddings via Reduction (Wu & Charikar, 2020)

Wu & Charikar (2020) take the alternative approach: reduce hyperbolic NNS to Euclidean NNS.

**Their "Spherical Shell" algorithm:**
1. Split the Poincare ball into thin annuli (concentric shells).
2. Within each shell, distances approximate Euclidean distances (up to a scaling factor).
3. Run standard Euclidean ANN (e.g., LSH) on each shell separately.
4. Union the results.

**Drawback:** Requires calling the Euclidean NNS method multiple times (one per shell), increasing complexity compared to the native Euclidean algorithm. The Prokhorenkova et al. (2022) results show that direct graph-based approaches are superior.

**Citation:** Wu, X., Charikar, M. "Nearest Neighbor Search for Hyperbolic Embeddings." 2020.
URL: https://arxiv.org/abs/2009.00836

---

## 2. Triangle Inequality and Navigability in the Poincare Ball

### 2.1 The Poincare Distance IS a Metric

The Poincare distance satisfies all metric axioms:
- Non-negativity: d(u,v) >= 0, with equality iff u = v
- Symmetry: d(u,v) = d(v,u)
- **Triangle inequality: d(u,w) <= d(u,v) + d(v,w)**

This is critical because HNSW's greedy search correctness depends on the triangle inequality. With a valid metric, if a neighbor `v` is closer to the target `q` than the current node `u`, then routing through `v` makes monotonic progress toward `q`.

### 2.2 How Constant Factors Differ

In Euclidean space, volumes grow polynomially: V(r) ~ r^d. In hyperbolic space, volumes grow **exponentially**: V(r) ~ e^{(d-1)r}.

This has a surprising **positive** effect on HNSW navigability:

- **Fewer hops needed:** Because hyperbolic volume grows exponentially, a ball of radius R contains exponentially many points. This means greedy routing converges faster -- in the dense regime, O(log n) hops suffice without needing explicit long-range connections.
- **Natural small-world structure:** Hyperbolic space inherently produces small-world network topologies. Points near the origin (high in the hierarchy) are naturally "long-range" links to distant regions. This is exactly the multi-scale connectivity HNSW tries to construct artificially.
- **The navigability property holds** -- and may be even better than in Euclidean space, because the geometry itself provides the multi-scale structure that HNSW's layered approach is designed to simulate.

### 2.3 Potential Issues

- **Near the boundary:** Points with ||x|| -> 1 have distances that grow without bound. The conformal factor lambda_x = 2/(1 - ||x||^2) diverges, meaning distance computations become numerically sensitive.
- **Non-uniform density:** Equal-volume regions contain very different numbers of points depending on distance from origin. The HNSW layer assignment heuristic (which uses a random exponential distribution) may need tuning to account for the natural hierarchy.

---

## 3. Existing Hyperbolic ANN Implementations

### 3.1 No Production-Ready Hyperbolic ANN Library Exists

There is **no** FAISS extension, hnswlib plugin, or production library that natively supports hyperbolic distance for ANN search. The landscape:

| System | Hyperbolic Support | Notes |
|--------|-------------------|-------|
| FAISS | No | Does not support custom distance functions. Built-in: L2, inner product, cosine only. |
| hnswlib | No | Supports L2, inner product, cosine. Custom distances require source modification. |
| Annoy | No | Euclidean, Manhattan, angular, dot product only. |
| ScaNN | No | Euclidean-focused with anisotropic quantization. |
| USearch | Partial | Supports custom metrics via C callbacks, could theoretically plug in Poincare distance. |
| KeOps | Research | Demonstrates K-NN on Poincare half-plane with custom GPU kernels. Not a full ANN system. |

### 3.2 Research Implementations

- **geoopt** (PyTorch): Full Poincare ball manifold operations (exp map, log map, parallel transport, geodesics). Not an ANN index -- a differential geometry library. URL: https://github.com/geoopt/geoopt
- **Prokhorenkova et al. (2022)**: Implemented graph-based NNS in hyperbolic space for their paper. Not released as a standalone library.
- **HoroPCA** (Chami et al., ICML 2021): Dimensionality reduction in hyperbolic space. Uses horospherical projections for PCA-like operations.

### 3.3 Implication for This Project

Building a native hyperbolic HNSW index for PostgreSQL would be **novel**. No existing system provides this. The closest approach would be to fork pgvector's HNSW implementation and replace the distance function -- but as we will see, more than just the distance function needs to change.

---

## 4. Neighbor Selection Heuristics (Algorithm 4 Modifications)

### 4.1 Original HNSW Algorithm 4

The Malkov & Yashunin (2018) neighbor selection heuristic (Algorithm 4) selects diverse neighbors by iterating through candidates in order of increasing distance and rejecting any candidate that is closer to an already-selected neighbor than to the target element. This prevents graph fragmentation from purely local clustering.

The key check is: for candidate `e` and already-selected neighbor `r`, if `dist(e, r) < dist(e, q)`, then `e` is "blocked" by `r` and not selected. This ensures angular diversity in the neighbor set.

**Citation:** Malkov, Y., Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2018.
URL: https://arxiv.org/abs/1603.09320

### 4.2 Required Modifications for Hyperbolic Space

The diversity heuristic fundamentally relies on the concept: "is candidate `e` well-separated from existing neighbor `r`?" This is a distance comparison, and **works with any metric**. The algorithm only needs:
1. A distance function d(x, y)
2. The triangle inequality (to ensure the diversity check is meaningful)

Both are satisfied by the Poincare metric. However, there are nuances:

**What changes:**
- **Distance scale is non-uniform.** Near the origin, distances between nearby points are small (roughly Euclidean). Near the boundary, even tiny Euclidean displacements correspond to large hyperbolic distances. The diversity heuristic's comparison `d(e,r) < d(e,q)` still works, but the **effective angular diversity** varies with position in the ball.
- **The "extended candidates" option** (exploring neighbors of candidates) traverses the graph. In hyperbolic space, this naturally explores hierarchically -- moving toward the origin explores "up" the hierarchy (broad connections), moving toward the boundary explores "down" (specific leaves).
- **Layer assignment:** HNSW assigns layers using `-ln(uniform()) * m_L`. In hyperbolic space, the natural hierarchy suggests that points near the origin (low radius, high in hierarchy) should appear on more layers (they're natural hubs). A modified formula could use the point's Poincare norm: `layer = floor(-ln(uniform()) * m_L * f(||x||))` where f(r) is larger for small r.

**What stays the same:**
- The greedy search algorithm (Algorithm 1-2): works unchanged with any metric.
- The insertion algorithm (Algorithm 3): works unchanged.
- The diversity heuristic logic (Algorithm 4): works unchanged, just with Poincare distance.

---

## 5. Quantization in Hyperbolic Space

### 5.1 The Problem with Standard Quantization

Product quantization (PQ) and scalar quantization (SQ) assume a **linear vector space**:
- PQ splits vectors into subvectors and independently quantizes each -- this assumes coordinates are independent and distances decompose additively across subspaces.
- SQ uniformly bins each dimension -- this assumes uniform metric scaling along each axis.

Neither assumption holds in the Poincare ball:
- The metric is **conformal** (g_x = lambda_x^2 * I), meaning it scales uniformly at each point but the scale depends on position.
- Distances do NOT decompose into independent per-dimension contributions.
- Splitting a Poincare vector into subvectors does not produce valid Poincare subvectors (the constraint ||x|| < 1 applies to the full vector, not subvectors).

### 5.2 HiHPQ: Hierarchical Hyperbolic Product Quantization (Xu et al., AAAI 2024)

This paper directly addresses product quantization in hyperbolic space:

1. **Decomposition in Lorentz model:** Instead of splitting the Poincare ball (problematic), they decompose the Lorentz model into M d-dimensional Lorentzian submanifolds, each with its own curvature parameter.
2. **Tangent space projection:** Project vectors onto tangent spaces (Euclidean), perform operations there, then map back via the exponential map.
3. **Codebook on the manifold:** Each codebook C^m = {c^m_k in H^d_{theta_m}} lives directly on the Lorentzian manifold. Assignment uses Lorentzian distance: d^2_L(a,b) = -2*theta - 2*<a,b>_L.
4. **Soft quantization:** Uses softmax-weighted aggregation of codewords rather than hard assignment, maintaining differentiability.

**Citation:** Xu et al. "HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval." AAAI 2024.
URL: https://arxiv.org/abs/2401.07212

### 5.3 Practical Options for a PostgreSQL Extension

Three approaches to quantization, in order of implementation complexity:

1. **No quantization initially.** Store full float64 Poincare coordinates. This is the simplest starting point and may be adequate for moderate-scale datasets (millions of vectors).

2. **Tangent-space quantization.** Map all points to the tangent space at the origin (via log_0), apply standard PQ/SQ there, then reconstruct approximate Poincare points (via exp_0) for distance computation. The distortion depends on how far points are from the origin.

3. **Native Lorentz quantization.** Implement HiHPQ-style decomposition in the Lorentz model. Highest accuracy but significant implementation effort.

**Recommendation:** Start with option 1 (no quantization). Add option 2 (tangent-space PQ) as a follow-up if scale demands it. Option 3 is research-grade and likely unnecessary for our use case.

---

## 6. Poincare vs Lorentz Model for Internal Representation

### 6.1 Numerical Stability Analysis (Mishne et al., ICML 2023)

Key findings from "The Numerical Stability of Hyperbolic Representation Learning":

| Property | Poincare Ball | Lorentz (Hyperboloid) |
|----------|--------------|----------------------|
| Max representable radius (float64) | ~38 | ~19 |
| Gradient scaling | O(delta^2 * ||grad f||) -- **vanishes** near boundary | O(||grad f||) -- **stable** |
| Boundary constraint | Hard: points collapse to boundary | Soft: constraint verification fails but coordinates persist |
| Distance computation | Division by (1 - ||x||^2)(1 - ||y||^2) -- **unstable** near boundary | Inner product <x,y>_L -- **stable** |

**Citation:** Mishne, G., et al. "The Numerical Stability of Hyperbolic Representation Learning." ICML 2023.
URL: https://arxiv.org/abs/2211.00181

### 6.2 The Recommended Hybrid Approach

**Store and compute in Lorentz internally; expose Poincare API externally.**

Rationale:
- **Distance computation is the hot path** in HNSW (called millions of times during index construction and search). The Lorentz distance `d(x,y) = arcosh(-<x,y>_L)` involves a single inner product followed by arcosh. The Poincare distance requires computing ||u-v||^2, (1-||u||^2), (1-||v||^2), their product, and the fraction -- with catastrophic cancellation near the boundary.
- **The Poincare model is more intuitive** for users: points are bounded in [0,1)^n, the origin is the "root" of the hierarchy, distance from origin indicates depth. The API should accept and return Poincare coordinates.
- **Conversion is cheap:** The isometry between models is a simple formula (see Section 7).

### 6.3 An Even Better Option: Euclidean Parametrization

Mishne et al. (2023) recommend representing hyperbolic points via unconstrained Euclidean vectors z in R^n, mapped to the ball via:
- Poincare: x = exp_0(z/2) = tanh(||z||/2) * z/||z||
- Lorentz: y = (cosh(||z||), sinh(||z||) * z/||z||)

This eliminates all representation constraints and boundary issues. For a database extension, this is worth considering: store z (unconstrained), compute distances via the Lorentz formula applied to the mapped coordinates.

---

## 7. Key Formulas

All formulas below use curvature parameter c > 0. The standard Poincare ball has c = 1. The ball is B_c^n = {x in R^n : c*||x||^2 < 1}.

### 7.1 Conformal Factor

```
lambda_x^c = 2 / (1 - c * ||x||^2)
```

The Riemannian metric tensor is: g_x = (lambda_x^c)^2 * I (conformal to Euclidean).

### 7.2 Poincare Distance

```
d_c(u, v) = (2 / sqrt(c)) * arctanh(sqrt(c) * || (-u) +_c v ||)
```

Equivalently (for c = 1):

```
d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
```

### 7.3 Mobius Addition

```
u +_c v = ((1 + 2c<u,v> + c||v||^2) * u + (1 - c||u||^2) * v)
           / (1 + 2c<u,v> + c^2 * ||u||^2 * ||v||^2)
```

Properties:
- Non-commutative: u +_c v != v +_c u (in general)
- Non-associative (but gyroassociative)
- Recovers Euclidean addition as c -> 0

### 7.4 Exponential Map (Tangent Space -> Ball)

Map tangent vector v in T_x(B_c^n) to a point on the ball:

```
exp_x^c(v) = x +_c ( tanh(sqrt(c) * lambda_x^c * ||v|| / 2) * v / (sqrt(c) * ||v||) )
```

At the origin (x = 0):

```
exp_0^c(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
```

### 7.5 Logarithmic Map (Ball -> Tangent Space)

Map point y on the ball to tangent vector at x:

```
log_x^c(y) = (2 / (sqrt(c) * lambda_x^c)) * arctanh(sqrt(c) * ||(-x) +_c y||)
              * ((-x) +_c y) / ||(-x) +_c y||
```

At the origin:

```
log_0^c(y) = arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
```

### 7.6 Parallel Transport

Transport tangent vector v from T_x to T_y along the geodesic:

**From origin to point y (simplified):**

```
P_{0->y}^c(v) = v * (lambda_0^c / lambda_y^c) = v * (1 - c||y||^2) / 2
```

Note: geoopt uses the convention P_{0->y}(v) = v * (1 + kappa * ||y||^2) where kappa = -c (negative curvature sign convention).

**General formula (x to y):**

```
P_{x->y}(v) = v - (<log_x(y), v>_x / d^2(x,y)) * (log_x(y) + log_y(x))
```

where <.,.>_x is the Riemannian inner product at x: <u,v>_x = (lambda_x^c)^2 * <u,v>_E.

### 7.7 Poincare Midpoint (Einstein Gyromidpoint)

The weighted midpoint of points {x_1, ..., x_n} with weights {alpha_1, ..., alpha_n}:

```
midpoint_c(x_1,...,x_n) = (1/2) *_c ( sum_i(alpha_i * lambda_{x_i}^c * x_i)
                                       / sum_i(|alpha_i| * (lambda_{x_i}^c - 1)) )
```

where `*_c` is Mobius scalar multiplication:

```
t *_c x = (1/sqrt(c)) * tanh(t * arctanh(sqrt(c) * ||x||)) * x / ||x||
```

For two points with equal weights (the simple midpoint):

```
midpoint(u, v) = (1/2) *_c ( (lambda_u * u + lambda_v * v) / (lambda_u + lambda_v - 2) )
```

where lambda_x = 2 / (1 - c||x||^2).

**Practical note:** The Einstein midpoint is most naturally computed in the Klein model. Convert Poincare -> Klein, take weighted Euclidean average with Lorentz factors as weights, convert Klein -> Poincare:

```
Klein:    x_K = 2 * x_P / (1 + ||x_P||^2)
Midpoint: m_K = sum(gamma_i * alpha_i * x_K_i) / sum(gamma_i * alpha_i)
          where gamma_i = 1 / sqrt(1 - ||x_K_i||^2)
Back:     m_P = m_K / (1 + sqrt(1 - ||m_K||^2))
```

### 7.8 Model Conversion (Poincare <-> Lorentz)

**Poincare to Lorentz:**

```
phi(x) = ( (1 + ||x||^2) / (1 - ||x||^2),  2*x / (1 - ||x||^2) )
```

(First component is the time coordinate y_0, remaining are spatial.)

**Lorentz to Poincare:**

```
psi(y) = y_{1:n} / (1 + y_0)
```

**Lorentz distance:**

```
d_L(x, y) = arcosh(-<x, y>_L)
where <x, y>_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n  (Minkowski inner product)
```

---

## 8. Synthesis: Architecture Recommendations

### 8.1 HNSW Feasibility: CONFIRMED

HNSW works in hyperbolic space with **minimal algorithmic modifications**:
- The greedy search, insertion, and neighbor selection algorithms work unchanged -- they only require a distance function satisfying the triangle inequality.
- The Poincare metric IS a metric (satisfies triangle inequality).
- Theoretical analysis (Prokhorenkova et al., 2022) shows graph-based NNS may be **more** efficient in hyperbolic space than Euclidean, due to exponential volume growth enabling faster greedy convergence.

### 8.2 What Must Change vs. What Stays the Same

| Component | Change Needed? | Details |
|-----------|---------------|---------|
| Distance function | YES | Replace L2/cosine with Poincare or Lorentz distance |
| Greedy search (Alg 1-2) | NO | Works with any metric |
| Insertion (Alg 3) | NO | Works with any metric |
| Neighbor selection (Alg 4) | NO | Diversity heuristic works with any metric |
| Layer assignment | MAYBE | Could benefit from radius-aware assignment |
| Vector storage | YES | Need Poincare or Lorentz coordinates |
| Quantization | DEFER | Start without; add tangent-space PQ later |
| Internal distance computation | YES | Use Lorentz internally for numerical stability |

### 8.3 Recommended Internal Architecture

```
User API (SQL):  Poincare coordinates in, Poincare coordinates out
                 d_poincare(a, b) function exposed
                        |
                        v
Storage layer:   Lorentz coordinates (n+1 dimensions for n-dim Poincare)
                 Converted on INSERT, converted back on SELECT
                        |
                        v
HNSW index:      Lorentz inner product for distance
                 arcosh(-<x,y>_L) -- numerically stable
                 Standard HNSW algorithms, no modifications
```

### 8.4 Key Risks

1. **Numerical precision near boundary:** Even with Lorentz, arcosh of values very close to 1 loses precision. May need `arcosh(x) = log(x + sqrt(x^2 - 1))` with careful implementation.
2. **Dimensionality overhead:** Lorentz representation adds 1 dimension (n-dim Poincare becomes (n+1)-dim Lorentz). For 768-dim embeddings, this is negligible.
3. **No existing benchmarks:** We would need to create our own benchmark suite for hyperbolic ANN quality (recall@k, QPS).
4. **Curvature parameter:** The extension should support variable curvature c, not just c=1. Different embedding methods produce different curvatures.

---

## References

1. Prokhorenkova, L., Shekhovtsov, A. "Graph-based Nearest Neighbor Search in Hyperbolic Spaces." ICLR 2022.
   https://openreview.net/forum?id=USIgIY6TNDe

2. Malkov, Y., Yashunin, D. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." IEEE TPAMI, 2018.
   https://arxiv.org/abs/1603.09320

3. Wu, X., Charikar, M. "Nearest Neighbor Search for Hyperbolic Embeddings." 2020.
   https://arxiv.org/abs/2009.00836

4. Ganea, O., Becigneul, G., Hofmann, T. "Hyperbolic Neural Networks." NeurIPS 2018.
   https://arxiv.org/abs/1805.09112

5. Shimizu, R., Mukuta, Y., Harada, T. "Hyperbolic Neural Networks++." ICLR 2021.
   https://arxiv.org/abs/2006.08210

6. Mishne, G., et al. "The Numerical Stability of Hyperbolic Representation Learning." ICML 2023.
   https://arxiv.org/abs/2211.00181

7. Xu, Z., et al. "HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval." AAAI 2024.
   https://arxiv.org/abs/2401.07212

8. Nickel, M., Kiela, D. "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry." ICML 2018.
   https://arxiv.org/abs/1806.03417

9. Kochurov, M., et al. "Geoopt: Riemannian Optimization in PyTorch." JMLR, 2020.
   https://github.com/geoopt/geoopt

10. Ungar, A. "Gyrovector Spaces." (various publications on Mobius gyrovector algebra)
    https://en.wikipedia.org/wiki/Gyrovector_space
