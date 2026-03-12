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

-- ============================================================
-- Phase 2: Geometric primitives
-- ============================================================

-- Möbius addition: x ⊕ y in the Poincaré ball
CREATE FUNCTION hyp_mobius_add(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Hyperbolic midpoint (Einstein midpoint via conformal factors)
CREATE FUNCTION hyp_midpoint(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Exponential map: project tangent vector onto the ball from base point
CREATE FUNCTION hyp_exp_map(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Logarithmic map: tangent vector at base pointing toward target
CREATE FUNCTION hyp_log_map(vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Geodesic interpolation: walk fraction t along geodesic from a to b
CREATE FUNCTION hyp_lerp(vector, vector, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Entailment cone test: is child in parent's cone with aperture K?
CREATE FUNCTION hyp_cone_contains(vector, vector, float8) RETURNS boolean
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Busemann function: horocyclic "height" toward ideal boundary direction
CREATE FUNCTION hyp_busemann(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Parallel transport: move tangent vector from one base to another
CREATE FUNCTION hyp_parallel_transport(vector, vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Hyperbolic analogy: "a is to b as c is to ?"
CREATE FUNCTION hyp_analogy(vector, vector, vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Distance from point to geodesic defined by two points
CREATE FUNCTION hyp_dist_to_geodesic(vector, vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Ideal boundary direction: normalize to unit sphere
CREATE FUNCTION hyp_ideal_direction(vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
