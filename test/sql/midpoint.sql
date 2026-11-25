-- hyp_midpoint(a, b) must return the hyperbolic midpoint: the point on the geodesic
-- from a to b at half the distance from each end.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_hyperbolic;

-- The midpoint of a point with itself is the point.
SELECT poincare_distance(hyp_midpoint('[0.3,0.1]', '[0.3,0.1]'), '[0.3,0.1]') < 1e-5 AS self_midpoint;

-- Equidistant from both ends, at half the distance, and on the geodesic.
WITH p AS (SELECT '[0.5,0]'::vector AS a, '[-0.2,0.3]'::vector AS b),
     m AS (SELECT a, b, hyp_midpoint(a, b) AS m FROM p)
SELECT abs(poincare_distance(a, m) - poincare_distance(a, b) / 2) < 1e-5 AS half_from_a,
       abs(poincare_distance(m, b) - poincare_distance(a, b) / 2) < 1e-5 AS half_from_b
FROM m;

-- Symmetric in its arguments.
SELECT poincare_distance(hyp_midpoint('[0.5,0]', '[-0.2,0.3]'),
                         hyp_midpoint('[-0.2,0.3]', '[0.5,0]')) < 1e-5 AS symmetric;

-- Agrees with hyp_lerp at t = 0.5 and with the hyp_centroid aggregate of the two points.
SELECT poincare_distance(hyp_midpoint('[0.5,0]', '[-0.2,0.3]'),
                         hyp_lerp('[0.5,0]', '[-0.2,0.3]', 0.5)) < 1e-5 AS matches_lerp;
SELECT poincare_distance(hyp_midpoint('[0.5,0]', '[-0.2,0.3]'), c) < 1e-5 AS matches_centroid
FROM (SELECT hyp_centroid(v) AS c FROM (VALUES ('[0.5,0]'::vector), ('[-0.2,0.3]'::vector)) t(v)) s;

-- Near the boundary, in three dimensions.
WITH p AS (SELECT '[0.9,0,0.1]'::vector AS a, '[0,0.9,-0.1]'::vector AS b),
     m AS (SELECT a, b, hyp_midpoint(a, b) AS m FROM p)
SELECT abs(poincare_distance(a, m) - poincare_distance(m, b)) < 1e-4 AS boundary_equidistant,
       abs(poincare_distance(a, m) + poincare_distance(m, b) - poincare_distance(a, b)) < 1e-4 AS boundary_on_geodesic
FROM m;
