#include "hyperbolic.h"

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
