#include "hyperbolic.h"

/*
 * Einstein midpoint aggregate for computing hyperbolic centroids.
 *
 * Algorithm (per-row transition):
 *   1. Poincaré → Klein: k_i = 2·x_i / (1 + ||x_i||²)
 *   2. Lorentz factor: γ_i = 1 / sqrt(1 - ||k_i||²)
 *   3. Accumulate: weighted_sum += γ_i · k_i, gamma_sum += γ_i
 *
 * Finalize:
 *   1. k_mu = weighted_sum / gamma_sum
 *   2. Klein → Poincaré: mu = k_mu / (1 + sqrt(1 - ||k_mu||²))
 *   3. Project to ball (safety)
 *
 * State is stored as a single float8 array:
 *   [0] = dim, [1] = count, [2] = gamma_sum, [3..3+dim-1] = weighted_sum
 */

#include "utils/array.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

/* ============================================================
 * hyp_centroid_sfunc(state internal[], vector) → float8[]
 * Transition function: accumulate one vector into state.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_centroid_sfunc);
Datum
hyp_centroid_sfunc(PG_FUNCTION_ARGS)
{
	ArrayType  *state;
	Vector	   *v;
	Datum	   *state_data;
	int			state_len, dim, i;
	double		x_sq, gamma, k_sq;
	bool		is_null;

	v = PG_GETARG_VECTOR_P(1);
	dim = v->dim;

	/* Initialize state on first call */
	is_null = PG_ARGISNULL(0);
	if (is_null)
	{
		int			total = 3 + dim;
		Datum	   *elems = (Datum *) palloc(sizeof(Datum) * total);

		elems[0] = Float8GetDatum((double) dim);
		elems[1] = Float8GetDatum(0.0); /* count */
		elems[2] = Float8GetDatum(0.0); /* gamma_sum */
		for (i = 0; i < dim; i++)
			elems[3 + i] = Float8GetDatum(0.0);

		state = construct_array(elems, total, FLOAT8OID,
								sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);
		pfree(elems);
	}
	else
	{
		state = PG_GETARG_ARRAYTYPE_P(0);
	}

	/* Deconstruct state array */
	{
		bool   *nulls;
		int		nelems;

		deconstruct_array(state, FLOAT8OID, sizeof(float8),
						  FLOAT8PASSBYVAL, TYPALIGN_DOUBLE,
						  &state_data, &nulls, &nelems);
		state_len = nelems;
		pfree(nulls);
	}

	/* Compute ||x||² */
	x_sq = hyp_norm_sq(v->x, dim);
	if (x_sq >= 1.0 - POINC_EPS)
		x_sq = 1.0 - POINC_EPS;

	/* Poincaré → Klein: k_i = 2·x_i / (1 + ||x||²) */
	/* ||k||² = 4·||x||² / (1 + ||x||²)² */
	k_sq = 4.0 * x_sq / ((1.0 + x_sq) * (1.0 + x_sq));
	if (k_sq >= 1.0 - POINC_EPS)
		k_sq = 1.0 - POINC_EPS;

	/* Lorentz factor: γ = 1 / sqrt(1 - ||k||²) */
	gamma = 1.0 / sqrt(1.0 - k_sq);

	/* Update state */
	state_data[1] = Float8GetDatum(DatumGetFloat8(state_data[1]) + 1.0);
	state_data[2] = Float8GetDatum(DatumGetFloat8(state_data[2]) + gamma);

	for (i = 0; i < dim; i++)
	{
		double ki = 2.0 * (double) v->x[i] / (1.0 + x_sq);
		double old = DatumGetFloat8(state_data[3 + i]);
		state_data[3 + i] = Float8GetDatum(old + gamma * ki);
	}

	/* Reconstruct array */
	state = construct_array(state_data, state_len, FLOAT8OID,
							sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);

	PG_RETURN_ARRAYTYPE_P(state);
}

/* ============================================================
 * hyp_centroid_ffunc(state float8[]) → vector
 * Final function: convert accumulated state to Poincaré point.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_centroid_ffunc);
Datum
hyp_centroid_ffunc(PG_FUNCTION_ARGS)
{
	ArrayType  *state;
	Datum	   *state_data;
	bool	   *nulls;
	int			nelems, dim, i;
	double		gamma_sum, k_mu_sq, denom;
	Vector	   *result;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = PG_GETARG_ARRAYTYPE_P(0);
	deconstruct_array(state, FLOAT8OID, sizeof(float8),
					  FLOAT8PASSBYVAL, TYPALIGN_DOUBLE,
					  &state_data, &nulls, &nelems);

	dim = (int) DatumGetFloat8(state_data[0]);
	gamma_sum = DatumGetFloat8(state_data[2]);

	if (gamma_sum < 1e-15)
		PG_RETURN_NULL();

	/* k_mu = weighted_sum / gamma_sum */
	/* Then Klein → Poincaré: mu = k_mu / (1 + sqrt(1 - ||k_mu||²)) */
	k_mu_sq = 0.0;
	result = HypInitVector(dim);

	for (i = 0; i < dim; i++)
	{
		double ki = DatumGetFloat8(state_data[3 + i]) / gamma_sum;
		k_mu_sq += ki * ki;
		result->x[i] = (float) ki; /* temporary: store Klein coords */
	}

	if (k_mu_sq >= 1.0 - POINC_EPS)
		k_mu_sq = 1.0 - POINC_EPS;

	denom = 1.0 + sqrt(1.0 - k_mu_sq);
	for (i = 0; i < dim; i++)
		result->x[i] = (float)((double) result->x[i] / denom);

	project_to_ball(result->x, dim);

	pfree(state_data);
	pfree(nulls);
	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_centroid_combinefunc(state1, state2) → float8[]
 * Combine two partial states for parallel aggregation.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_centroid_combinefunc);
Datum
hyp_centroid_combinefunc(PG_FUNCTION_ARGS)
{
	ArrayType  *s1, *s2;
	Datum	   *d1, *d2;
	bool	   *n1, *n2;
	int			len1, len2, i;

	if (PG_ARGISNULL(0))
		PG_RETURN_DATUM(PG_GETARG_DATUM(1));
	if (PG_ARGISNULL(1))
		PG_RETURN_DATUM(PG_GETARG_DATUM(0));

	s1 = PG_GETARG_ARRAYTYPE_P(0);
	s2 = PG_GETARG_ARRAYTYPE_P(1);

	deconstruct_array(s1, FLOAT8OID, sizeof(float8),
					  FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &d1, &n1, &len1);
	deconstruct_array(s2, FLOAT8OID, sizeof(float8),
					  FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &d2, &n2, &len2);

	/* Add counts, gamma_sums, and weighted sums */
	for (i = 1; i < len1; i++)
		d1[i] = Float8GetDatum(DatumGetFloat8(d1[i]) + DatumGetFloat8(d2[i]));

	pfree(n1);
	pfree(n2);
	pfree(d2);

	s1 = construct_array(d1, len1, FLOAT8OID,
						 sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);
	pfree(d1);
	PG_RETURN_ARRAYTYPE_P(s1);
}
