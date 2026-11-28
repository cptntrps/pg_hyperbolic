-- pg_hyperbolic 0.1.0 -> 0.1.1
-- The two IVFFlat operator classes had no support function 3 (the
-- k-means distance), so CREATE INDEX ... USING ivfflat failed with
-- "missing support function 3". Add it: the same hyperbolic distance.
\echo Use "ALTER EXTENSION pg_hyperbolic UPDATE TO '0.1.1'" to load this file. \quit

ALTER OPERATOR FAMILY vector_poincare_ivfflat_ops USING ivfflat
    ADD FUNCTION 3 (vector, vector) poincare_distance(vector, vector);

ALTER OPERATOR FAMILY vector_lorentz_ivfflat_ops USING ivfflat
    ADD FUNCTION 3 (vector, vector) poincare_distance_lorentz(vector, vector);
