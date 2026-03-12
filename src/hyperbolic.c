#include "hyperbolic.h"
#include "varatt.h"

PG_MODULE_MAGIC;

/* Allocate a new Vector (InitVector is not exported by pgvector) */
static Vector *
HypInitVector(int dim)
{
	Vector *result = (Vector *) palloc0(VECTOR_SIZE(dim));

	SET_VARSIZE(result, VECTOR_SIZE(dim));
	result->dim = dim;
	result->unused = 0;
	return result;
}

/* ---- Core distance computation ---- */

HYP_TARGET_CLONES
static double
PoincareBallDistance(int dim, float *ax, float *bx)
{
	double x_sq = 0.0, y_sq = 0.0, diff_sq = 0.0;

	for (int i = 0; i < dim; i++)
	{
		double xi = (double) ax[i];
		double yi = (double) bx[i];
		double d  = xi - yi;
		x_sq    += xi * xi;
		y_sq    += yi * yi;
		diff_sq += d * d;
	}

	double denom;
	double arg;

	/* Clamp norms to ball interior */
	if (x_sq >= 1.0 - POINC_EPS)
		x_sq = 1.0 - POINC_EPS;
	if (y_sq >= 1.0 - POINC_EPS)
		y_sq = 1.0 - POINC_EPS;

	denom = (1.0 - x_sq) * (1.0 - y_sq);
	arg   = 1.0 + 2.0 * diff_sq / denom;

	return stable_arcosh(arg);
}

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

	for (int i = 0; i < v->dim; i++)
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

	for (int i = 0; i < v->dim; i++)
	{
		double xi = (double) v->x[i];
		norm_sq += xi * xi;
	}
	double norm;

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

	for (int i = 0; i < v->dim; i++)
	{
		double xi = (double) v->x[i];
		norm_sq += xi * xi;
	}
	PG_RETURN_BOOL(norm_sq < 1.0);
}
