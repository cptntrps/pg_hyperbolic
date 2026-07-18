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
-- Alternative distance: Lorentz (hyperboloid) model
-- Converts Poincaré→Lorentz internally, uses Minkowski IP.
-- No boundary-singular denominator. Transparent to ANN index.
-- ============================================================
CREATE FUNCTION poincare_distance_lorentz(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Operator: <~> via Lorentz model (same symbol, different function)
-- Users choose which operator class to use at index creation time.
CREATE OPERATOR <%> (
    LEFTARG = vector, RIGHTARG = vector,
    PROCEDURE = poincare_distance_lorentz,
    COMMUTATOR = '<%>'
);

-- HNSW Operator Class (Lorentz model distance)
CREATE OPERATOR CLASS vector_lorentz_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <%> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance_lorentz(vector, vector);

-- ============================================================
-- IVFFlat Operator Classes
-- Same distance functions, registered for IVFFlat index method.
-- Centroid computation uses the hyp_centroid aggregate.
-- ============================================================
CREATE OPERATOR CLASS vector_poincare_ivfflat_ops
    FOR TYPE vector USING ivfflat AS
    OPERATOR 1 <~> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance(vector, vector);

CREATE OPERATOR CLASS vector_lorentz_ivfflat_ops
    FOR TYPE vector USING ivfflat AS
    OPERATOR 1 <%> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 poincare_distance_lorentz(vector, vector);

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

-- Conformal factor: λ = 2/(1-||x||²). Grows near boundary.
CREATE FUNCTION hyp_conformal_factor(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Map Euclidean vector into Poincaré ball: exp_0(v) = tanh(||v||·scale/2)·v/||v||
CREATE FUNCTION hyp_from_euclidean(vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hyp_from_euclidean(vector, float8) RETURNS vector
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

-- ============================================================
-- Phase 3: Hyperbolic centroid aggregate
-- ============================================================

-- Internal transition/final/combine functions
CREATE FUNCTION hyp_centroid_sfunc(float8[], vector) RETURNS float8[]
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION hyp_centroid_ffunc(float8[]) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION hyp_centroid_combinefunc(float8[], float8[]) RETURNS float8[]
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE PARALLEL SAFE;

-- Einstein midpoint aggregate: computes hyperbolic centroid via Klein model
CREATE AGGREGATE hyp_centroid(vector) (
    SFUNC = hyp_centroid_sfunc,
    STYPE = float8[],
    FINALFUNC = hyp_centroid_ffunc,
    COMBINEFUNC = hyp_centroid_combinefunc,
    PARALLEL = SAFE
);

-- ============================================================
-- Variable curvature operations
-- Curvature c > 0 means sectional curvature = -c, ball radius = 1/√c.
-- For c = 1.0, results match the standard (c=1) functions above.
-- ============================================================

-- Distance with variable curvature
CREATE FUNCTION poincare_distance_c(vector, vector, float8) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Möbius addition with variable curvature
CREATE FUNCTION hyp_mobius_add_c(vector, vector, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Hyperbolic radius with variable curvature
CREATE FUNCTION hyp_radius_c(vector, float8) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Exponential map with variable curvature
CREATE FUNCTION hyp_exp_map_c(vector, vector, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Logarithmic map with variable curvature
CREATE FUNCTION hyp_log_map_c(vector, vector, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Project to ball of radius 1/√c
CREATE FUNCTION hyp_project_c(vector, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Geodesic interpolation with variable curvature
CREATE FUNCTION hyp_lerp_c(vector, vector, float8, float8) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================
-- Precision-adaptive boundary epsilon
-- Returns the recommended POINC_EPS for a given storage precision.
-- ============================================================
CREATE FUNCTION hyp_boundary_epsilon(text) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
