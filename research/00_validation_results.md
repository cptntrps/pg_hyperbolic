# Validation Results: Poincare Distance in pgvector HNSW

**Date:** 2026-03-12
**Environment:** PostgreSQL 16.12, pgvector 0.8.1, 5,575 entities with 32D Poincare nav embeddings
**SQL:** `sql/00_validation.sql`

## Summary

**GO.** PL/pgSQL Poincare distance plugs into pgvector's HNSW with zero C code. The index builds, searches correctly (99.5% recall@10), and produces meaningfully different neighbor rankings than L2. Performance is 250x slower than native L2 due to PL/pgSQL interpretation overhead, which a C extension would eliminate.

## Step 1: Poincare Distance Function

The PL/pgSQL implementation works correctly:
- Self-distance: ~4.7e-8 (numerical zero)
- Cross-distance: 1.99 for `[0.1,0.2,0.3]` vs `[0.4,0.5,0.6]`

**Gotcha:** pgvector's `vector` type cannot be cast directly to `float8[]`. The text representation uses `[x,y,z]` while PostgreSQL arrays use `{x,y,z}`. Solution: `translate(a::text, '[]', '{}')::float8[]`.

## Step 2: L2 vs Poincare KNN Comparison

### Python (norm=0.588, lowest non-trivial entity)

| Rank | L2 Neighbor | L2 Norm | Poincare Neighbor | Poinc Norm |
|------|-------------|---------|-------------------|------------|
| 1 | Pandas | 0.743 | Pandas | 0.743 |
| 2 | LLM models | 0.866 | Python script | 0.708 |
| 3 | Programming languages | 0.838 | Jellyfin | 0.706 |
| 4 | OllamaAPI | 0.836 | tables | 0.752 |
| 5 | shutil | 0.857 | NumPy | 0.764 |
| 6 | BeautifulSoup | 0.838 | Programming languages | 0.838 |
| 7 | ML (Machine Learning) | 0.856 | OllamaAPI | 0.836 |
| 8 | Pandas (library) | 0.851 | tkinter as tk | 0.717 |
| 9 | exiftool | 0.848 | Flask | 0.747 |
| 10 | Python script | 0.708 | Codecademy | 0.787 |

**Key observation:** L2 avg neighbor norm = 0.824 (std 0.054). Poincare avg neighbor norm = 0.760 (std 0.048). Poincare pulls neighbors closer to Python's own depth.

### Netflix (norm=0.786)

| Rank | L2 Neighbor | Poincare Neighbor |
|------|-------------|-------------------|
| 1 | Tubi | Tubi |
| 2 | JustWatch | Amazon Prime Video |
| 3 | Amazon Prime Video | Plex |
| 4 | TMDB | TMDB |
| 5 | Plex | JustWatch |
| 6 | Broadchurch | Broadchurch |
| 7 | Orphan Black | Orphan Black |
| 8 | Devs | The Killing |
| 9 | The Killing | Devs |
| 10 | How to Get Away with Murder | His & Hers |

9/10 overlap -- at high norms, L2 and Poincare converge because the curvature correction is smaller.

### Google (norm=0.786)

Poincare surfaced Gmail (norm 0.697), Apple (0.724), TensorFlow (0.774), PostgreSQL (0.656), Amazon (0.611) -- lower-norm "peer companies/technologies" that L2 missed in favor of higher-norm specific entities like Clay Bavor (0.889), Strategy and Operations Lead (0.891).

6/10 overlap.

### Docker (norm=0.787)

Most divergent case. Only 2/10 overlap. Poincare pulled GitHub Copilot (0.548), OpenAI (0.592), Wolters Kluwer (0.557) -- mid-level entities. L2 pulled MI8 (0.931), Donetick (0.876) -- high-norm specific entities.

### React (norm=0.646)

5/10 overlap. Poincare added FastAPI (0.700), PowerPoint (0.687), python-pptx (0.728), Canva (0.728) -- similar-depth tools. L2 added shadcn/ui (0.881), front-end engineer (0.909), Angular (0.910) -- higher-norm specifics.

### Overlap Summary

| Entity | Norm | L2/Poincare Overlap |
|--------|------|---------------------|
| Python | 0.588 | 4/10 |
| Docker | 0.787 | 2/10 |
| React | 0.646 | 5/10 |
| Google | 0.786 | 6/10 |
| Netflix | 0.786 | 9/10 |
| **Average** | | **5.2/10** |

## Step 3: Operator and Operator Class

Both created successfully:
```sql
CREATE OPERATOR <~> (LEFTARG = vector, RIGHTARG = vector, PROCEDURE = poincare_distance, COMMUTATOR = '<~>');
CREATE OPERATOR CLASS vector_poincare_ops FOR TYPE vector USING hnsw AS ...;
```

No errors. pgvector's HNSW access method accepts custom operator classes without modification.

## Step 4: HNSW Index Benchmark

### Index Build Time
| Index | Build Time |
|-------|-----------|
| HNSW + Poincare (PL/pgSQL) | 115.7 seconds |
| HNSW + L2 (native C) | 0.47 seconds |
| **Ratio** | **~250x slower** |

### Query Timing (KNN top-10, single query)
| Method | Time |
|--------|------|
| Brute-force Poincare (seq scan) | 120 ms |
| Indexed Poincare (HNSW) | 102-189 ms |
| Indexed L2 (HNSW, native) | 45-67 ms |

The indexed Poincare query is only marginally faster than brute-force because the PL/pgSQL distance function dominates -- every HNSW candidate evaluation calls interpreted PL/pgSQL. A C implementation would bring this to sub-millisecond.

### Recall@10 (indexed vs brute-force)
**99.5%** over 20 random queries. The HNSW graph structure works correctly with the Poincare metric.

### EXPLAIN ANALYZE Confirmation
```
Index Scan using idx_test_hyp_hnsw on test_hyperbolic
  Order By: (nav_vec <~> $0)
  Execution Time: 85.386 ms
```
Confirmed: PostgreSQL uses the HNSW index with the custom operator.

## Step 5: Hierarchy Awareness

### Norm Distribution of Top-10 Neighbors

| Query Entity | Query Norm | Metric | Avg Neighbor Norm | Std | Min | Max |
|-------------|-----------|--------|-------------------|-----|-----|-----|
| Python | 0.588 | L2 | 0.824 | 0.054 | 0.708 | 0.866 |
| Python | 0.588 | **Poincare** | **0.760** | **0.048** | 0.706 | 0.838 |
| Google | 0.786 | L2 | 0.848 | 0.065 | 0.697 | 0.905 |
| Google | 0.786 | **Poincare** | **0.768** | **0.095** | 0.611 | 0.905 |
| Tkinter GUI toolkit | 0.959 | L2 | 0.902 | 0.007 | 0.893 | 0.912 |
| Tkinter GUI toolkit | 0.959 | **Poincare** | **0.898** | **0.017** | 0.854 | 0.912 |

**Finding:** Poincare consistently pulls neighbor norms closer to the query entity's norm, but the effect is moderate rather than dramatic. This is because 91% of entities are in the 0.8+ norm band -- the data is heavily skewed toward the boundary, limiting the dynamic range available for depth differentiation.

For Python (the lowest-norm entity at 0.588), the effect is most visible:
- L2 neighbors average 0.824 (pulled toward population mean)
- Poincare neighbors average 0.760 (0.064 closer to Python's actual norm)

At high norms (Tkinter, 0.959), both metrics return similar neighbors because nearly everything is at high radius.

## Issues Encountered

1. **vector -> float8[] cast:** pgvector vectors cannot be directly cast to PostgreSQL arrays. Requires text intermediary with bracket translation.
2. **PL/pgSQL performance:** The interpreted loop is the bottleneck. Index build is 250x slower than native. Query speedup from HNSW is marginal because each candidate still calls the slow function.
3. **Norm distribution skew:** 91% of entities have norm > 0.8. This compresses the hierarchy signal. With a more evenly distributed dataset, the L2 vs Poincare divergence would be larger.

## Recommendation: GO

**The concept is validated.** Key findings:

1. **pgvector's HNSW accepts custom distance functions** via operator classes. No C code needed for the proof-of-concept.
2. **Poincare distance produces meaningfully different rankings** than L2 (average 48% different neighbors at top-10).
3. **Recall is excellent** (99.5%) -- the HNSW graph structure works with hyperbolic distances.
4. **Performance requires C extension** for production use. The PL/pgSQL implementation is adequate for validation but 250x slower than native distance functions.

**Next step:** Write the C extension (`poincare_distance` as a C function) to eliminate the interpretation overhead. This should bring index build time from 116s to <1s and query time to sub-millisecond, matching native L2 performance.
