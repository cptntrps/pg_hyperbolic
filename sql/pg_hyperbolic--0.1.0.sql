-- pg_hyperbolic: Poincaré distance for pgvector
-- Requires: pgvector (CREATE EXTENSION vector)

-- ============================================================
-- Distance function
-- ============================================================
CREATE FUNCTION poincare_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================
-- Operator: <~> for Poincaré distance
-- ============================================================
CREATE OPERATOR <~> (
    LEFTARG = vector, RIGHTARG = vector,
    PROCEDURE = poincare_distance,
    COMMUTATOR = '<~>'
);

-- ============================================================
-- HNSW Operator Class
-- ============================================================
CREATE OPERATOR CLASS vector_poincare_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <~> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector);

-- ============================================================
-- Utility functions
-- ============================================================
CREATE FUNCTION poincare_norm(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_radius(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_project(vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_check_ball(vector) RETURNS boolean
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
