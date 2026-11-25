#include "hyperbolic.h"
#include "utils/builtins.h"	/* text_to_cstring */

PG_MODULE_MAGIC;

/* ---- PG-registered functions ---- */

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(poincare_distance);
Datum
poincare_distance(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);

	HypCheckDims(a, b);

	PG_RETURN_FLOAT8(PoincareBallDistance(a->dim, a->x, b->x));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(poincare_norm);
Datum
poincare_norm(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double  norm_sq = 0.0;
	int i;

	for (i = 0; i < v->dim; i++)
	{
		double xi = (double) v->x[i];
		norm_sq += xi * xi;
	}
	PG_RETURN_FLOAT8(sqrt(norm_sq));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_radius);
Datum
hyp_radius(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double  norm_sq = 0.0;
	double  norm;
	int i;

	for (i = 0; i < v->dim; i++)
	{
		double xi = (double) v->x[i];
		norm_sq += xi * xi;
	}

	norm = sqrt(norm_sq);
	if (norm >= 1.0 - POINC_EPS)
		norm = 1.0 - POINC_EPS;

	/* d(0, x) = 2 * atanh(||x||) */
	PG_RETURN_FLOAT8(2.0 * atanh(norm));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_project);
Datum
hyp_project(PG_FUNCTION_ARGS)
{
	Vector *v      = PG_GETARG_VECTOR_P(0);
	Vector *result = HypInitVector(v->dim);

	memcpy(result->x, v->x, sizeof(float) * v->dim);
	project_to_ball(result->x, v->dim);

	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_from_euclidean(vector [, scale]) → vector
 * Map a Euclidean vector into the Poincaré ball via exp_0.
 * exp_0(v) = tanh(||v|| * scale / 2) * v / ||v||
 * Default scale = 1.0. Smaller scale = points closer to origin.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_from_euclidean);
Datum
hyp_from_euclidean(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double  scale = 1.0;
	double  norm, factor;
	Vector *result;
	int i;

	if (PG_NARGS() >= 2 && !PG_ARGISNULL(1))
		scale = PG_GETARG_FLOAT8(1);

	norm = sqrt(hyp_norm_sq(v->x, v->dim));
	result = HypInitVector(v->dim);

	if (norm < 1e-15)
	{
		/* Zero vector maps to origin */
		PG_RETURN_VECTOR_P(result);
	}

	factor = tanh(norm * scale / 2.0) / norm;
	for (i = 0; i < v->dim; i++)
		result->x[i] = (float)(factor * (double) v->x[i]);

	project_to_ball(result->x, v->dim);
	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_conformal_factor(vector) → float8
 * λ_x = 2 / (1 - ||x||²)
 * Grows as entity approaches boundary. Used for radius-dependent
 * threshold scaling (e.g., established entities resist changes more).
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_conformal_factor);
Datum
hyp_conformal_factor(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);

	PG_RETURN_FLOAT8(hyp_lambda(v->x, v->dim));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_check_ball);
Datum
hyp_check_ball(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double  norm_sq = 0.0;
	int i;

	for (i = 0; i < v->dim; i++)
	{
		double xi = (double) v->x[i];
		norm_sq += xi * xi;
	}
	PG_RETURN_BOOL(norm_sq < 1.0);
}

/* ============================================================
 * poincare_distance_lorentz(vector, vector) → float8
 * Alternative distance via Lorentz (hyperboloid) model.
 * Converts Poincaré→Lorentz internally, computes Minkowski IP.
 * No boundary-singular denominator. Transparent to ANN index.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(poincare_distance_lorentz);
Datum
poincare_distance_lorentz(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);

	HypCheckDims(a, b);

	PG_RETURN_FLOAT8(LorentzDistance(a->dim, a->x, b->x));
}

/* ============================================================
 * hyp_boundary_epsilon(text) → float8
 * Returns the recommended boundary stabilization epsilon for a
 * given storage precision. Adapts POINC_EPS to the precision of
 * the intermediate arithmetic relative to storage precision.
 *
 * Supported: 'float16', 'float32', 'float64'
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_boundary_epsilon);
Datum
hyp_boundary_epsilon(PG_FUNCTION_ARGS)
{
	text   *prec = PG_GETARG_TEXT_PP(0);
	char   *p = text_to_cstring(prec);
	double  eps;

	if (strcmp(p, "float16") == 0)
		eps = 1e-3;        /* float16: ~3.3 decimal digits */
	else if (strcmp(p, "float32") == 0)
		eps = 1e-7;        /* float32 stored, float64 intermediate */
	else if (strcmp(p, "float64") == 0)
		eps = 1e-15;       /* float64 stored, float64 intermediate */
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("unknown precision: %s (use float16, float32, or float64)", p)));

	pfree(p);
	PG_RETURN_FLOAT8(eps);
}
