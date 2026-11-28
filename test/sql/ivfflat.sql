-- The two IVFFlat operator classes build an index and the index returns
-- the exact nearest rows when the scan probes every list.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_hyperbolic;

-- 1000 deterministic points inside the Poincaré ball (norm < 0.9).
CREATE TABLE ivf_t (id int PRIMARY KEY, v vector(3));
INSERT INTO ivf_t
SELECT g, hyp_project(('[' || 0.5 * sin(g) || ',' || 0.5 * cos(3 * g) || ',' || 0.4 * sin(7 * g) || ']')::vector)
FROM generate_series(1, 1000) g;

CREATE INDEX ivf_t_p ON ivf_t USING ivfflat (v vector_poincare_ivfflat_ops) WITH (lists = 10);
CREATE INDEX ivf_t_l ON ivf_t USING ivfflat (v vector_lorentz_ivfflat_ops) WITH (lists = 10);

-- Exact answers with no index.
SET enable_indexscan = off;
SET enable_bitmapscan = off;
CREATE TEMP TABLE exact_p AS SELECT array_agg(id ORDER BY d, id) AS ids FROM (SELECT id, v <~> '[0.1,0.2,-0.1]' AS d FROM ivf_t ORDER BY 2, 1 LIMIT 10) s;
CREATE TEMP TABLE exact_l AS SELECT array_agg(id ORDER BY d, id) AS ids FROM (SELECT id, v <%> '[0.1,0.2,-0.1]' AS d FROM ivf_t ORDER BY 2, 1 LIMIT 10) s;
RESET enable_indexscan;
RESET enable_bitmapscan;

-- Index answers with every list probed.
SET enable_seqscan = off;
SET ivfflat.probes = 10;
EXPLAIN (COSTS OFF) SELECT id FROM ivf_t ORDER BY v <~> '[0.1,0.2,-0.1]' LIMIT 10;
EXPLAIN (COSTS OFF) SELECT id FROM ivf_t ORDER BY v <%> '[0.1,0.2,-0.1]' LIMIT 10;
SELECT (SELECT array_agg(id ORDER BY d, id) FROM (SELECT id, v <~> '[0.1,0.2,-0.1]' AS d FROM ivf_t ORDER BY v <~> '[0.1,0.2,-0.1]' LIMIT 10) s)
       = (SELECT ids FROM exact_p) AS poincare_exact;
SELECT (SELECT array_agg(id ORDER BY d, id) FROM (SELECT id, v <%> '[0.1,0.2,-0.1]' AS d FROM ivf_t ORDER BY v <%> '[0.1,0.2,-0.1]' LIMIT 10) s)
       = (SELECT ids FROM exact_l) AS lorentz_exact;

-- A row inserted after the build is found.
INSERT INTO ivf_t VALUES (1001, '[0.1,0.2,-0.1]');
SELECT id FROM ivf_t ORDER BY v <~> '[0.1,0.2,-0.1]' LIMIT 1;
SELECT id FROM ivf_t ORDER BY v <%> '[0.1,0.2,-0.1]' LIMIT 1;
RESET ivfflat.probes;
RESET enable_seqscan;

-- Fewer rows than lists (pgvector adds random centers) and an empty table.
CREATE TABLE ivf_small (id int, v vector(2));
INSERT INTO ivf_small VALUES (1, '[0.1,0.1]'), (2, '[-0.5,0.2]'), (3, '[0.3,-0.6]');
CREATE INDEX ON ivf_small USING ivfflat (v vector_poincare_ivfflat_ops) WITH (lists = 10);
CREATE INDEX ON ivf_small USING ivfflat (v vector_lorentz_ivfflat_ops) WITH (lists = 10);
SET enable_seqscan = off;
SET ivfflat.probes = 10;
SELECT id FROM ivf_small ORDER BY v <~> '[0.3,-0.5]' LIMIT 1;
SELECT id FROM ivf_small ORDER BY v <%> '[0.3,-0.5]' LIMIT 1;
RESET ivfflat.probes;
RESET enable_seqscan;
CREATE TABLE ivf_empty (v vector(2));
CREATE INDEX ON ivf_empty USING ivfflat (v vector_poincare_ivfflat_ops) WITH (lists = 4);
CREATE INDEX ON ivf_empty USING ivfflat (v vector_lorentz_ivfflat_ops) WITH (lists = 4);

DROP TABLE ivf_t, ivf_small, ivf_empty;
