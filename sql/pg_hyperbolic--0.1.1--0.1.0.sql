-- pg_hyperbolic 0.1.1 -> 0.1.0 (reversal of 0.1.0--0.1.1)
-- Removes support function 3 from the two IVFFlat operator classes.
-- An IVFFlat index on either class needs that function, so the
-- downgrade refuses while such an index exists.
\echo Use "ALTER EXTENSION pg_hyperbolic UPDATE TO '0.1.0'" to load this file. \quit

DO $$
DECLARE
    idx text;
BEGIN
    SELECT string_agg(i.indexrelid::regclass::text, ', ') INTO idx
      FROM pg_catalog.pg_index i
      JOIN pg_catalog.pg_opclass c ON c.oid = ANY (i.indclass::oid[])
      JOIN pg_catalog.pg_am a ON a.oid = c.opcmethod
     WHERE a.amname = 'ivfflat'
       AND c.opcname IN ('vector_poincare_ivfflat_ops', 'vector_lorentz_ivfflat_ops');
    IF idx IS NOT NULL THEN
        RAISE EXCEPTION 'pg_hyperbolic downgrade to 0.1.0 refused: IVFFlat indexes use the hyperbolic operator classes: %', idx;
    END IF;
END
$$;

ALTER OPERATOR FAMILY vector_poincare_ivfflat_ops USING ivfflat
    DROP FUNCTION 3 (vector, vector);

ALTER OPERATOR FAMILY vector_lorentz_ivfflat_ops USING ivfflat
    DROP FUNCTION 3 (vector, vector);
