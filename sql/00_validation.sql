-- Hyperbolic pgvector validation SQL
-- Poincare distance function + operator + HNSW operator class
-- Tested against PostgreSQL 16.12, pgvector 0.8.1
-- 2026-03-12

------------------------------------------------------------------------
-- Step 1: Poincare distance function (PL/pgSQL)
------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION poincare_distance(a vector, b vector) RETURNS float8 AS $$
DECLARE
    diff_sq float8 := 0;
    norm_a_sq float8 := 0;
    norm_b_sq float8 := 0;
    dim int;
    arr_a float8[];
    arr_b float8[];
BEGIN
    -- vector type text representation uses [x,y,...] but float8[] needs {x,y,...}
    arr_a := translate(a::text, '[]', '{}')::float8[];
    arr_b := translate(b::text, '[]', '{}')::float8[];
    dim := array_length(arr_a, 1);
    FOR i IN 1..dim LOOP
        diff_sq := diff_sq + (arr_a[i] - arr_b[i])^2;
        norm_a_sq := norm_a_sq + arr_a[i]^2;
        norm_b_sq := norm_b_sq + arr_b[i]^2;
    END LOOP;
    -- Clamp norms to avoid division by zero at Poincare ball boundary
    norm_a_sq := LEAST(norm_a_sq, 0.9999999999);
    norm_b_sq := LEAST(norm_b_sq, 0.9999999999);
    RETURN acosh(GREATEST(1.0 + 1e-15, 1.0 + 2.0 * diff_sq / ((1.0 - norm_a_sq) * (1.0 - norm_b_sq))));
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

------------------------------------------------------------------------
-- Step 2: Operator
------------------------------------------------------------------------
DROP OPERATOR IF EXISTS <~> (vector, vector);
CREATE OPERATOR <~> (
    LEFTARG = vector, RIGHTARG = vector,
    PROCEDURE = poincare_distance,
    COMMUTATOR = '<~>'
);

------------------------------------------------------------------------
-- Step 3: HNSW operator class (registers with pgvector's HNSW AM)
------------------------------------------------------------------------
-- NOTE: Will fail if already exists; DROP first if re-running
-- DROP OPERATOR CLASS IF EXISTS vector_poincare_ops USING hnsw;
CREATE OPERATOR CLASS vector_poincare_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <~> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector);

------------------------------------------------------------------------
-- Step 4: Test table + HNSW index (adapt to your data)
------------------------------------------------------------------------
DROP TABLE IF EXISTS test_hyperbolic;
CREATE TABLE test_hyperbolic AS
SELECT id, name, type,
       ('[' || array_to_string(poincare_nav, ',') || ']')::vector(32) as nav_vec,
       sqrt((SELECT sum(v*v) FROM unnest(poincare_nav) AS v)) as norm
FROM graph.nodes
WHERE poincare_nav IS NOT NULL AND expired_at IS NULL;

-- Build HNSW index with Poincare distance (slow: PL/pgSQL overhead)
CREATE INDEX idx_test_hyp_hnsw ON test_hyperbolic
USING hnsw (nav_vec vector_poincare_ops);

-- Build L2 index for comparison
CREATE INDEX idx_test_hyp_l2 ON test_hyperbolic
USING hnsw (nav_vec vector_l2_ops);

------------------------------------------------------------------------
-- Step 5: KNN queries
------------------------------------------------------------------------

-- Poincare KNN (will use HNSW index)
SELECT name, type, round(norm::numeric, 4) as norm,
       round((nav_vec <~> (SELECT nav_vec FROM test_hyperbolic
                           WHERE name = 'Python' AND type = 'technology' LIMIT 1))::numeric, 6) as poinc_dist
FROM test_hyperbolic
ORDER BY nav_vec <~> (SELECT nav_vec FROM test_hyperbolic
                      WHERE name = 'Python' AND type = 'technology' LIMIT 1)
LIMIT 10;

-- L2 KNN for comparison
SELECT name, type, round(norm::numeric, 4) as norm,
       round((nav_vec <-> (SELECT nav_vec FROM test_hyperbolic
                           WHERE name = 'Python' AND type = 'technology' LIMIT 1))::numeric, 6) as l2_dist
FROM test_hyperbolic
ORDER BY nav_vec <-> (SELECT nav_vec FROM test_hyperbolic
                      WHERE name = 'Python' AND type = 'technology' LIMIT 1)
LIMIT 10;

------------------------------------------------------------------------
-- Cleanup
------------------------------------------------------------------------
-- DROP TABLE IF EXISTS test_hyperbolic;
-- DROP OPERATOR CLASS IF EXISTS vector_poincare_ops USING hnsw;
-- DROP OPERATOR IF EXISTS <~> (vector, vector);
-- DROP FUNCTION IF EXISTS poincare_distance(vector, vector);
