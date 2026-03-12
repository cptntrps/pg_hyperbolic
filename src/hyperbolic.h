#ifndef PG_HYPERBOLIC_H
#define PG_HYPERBOLIC_H

#include "postgres.h"
#include "fmgr.h"
#include "varatt.h"
#include "extension/vector/vector.h"
#include <math.h>
#include <string.h>

/* Stability constants */
#define POINC_EPS           1e-7
#define ARCOSH_TAYLOR_THR   1e-8

/* Auto-vectorization (match pgvector pattern) */
#ifdef __x86_64__
#define HYP_TARGET_CLONES \
	__attribute__((target_clones("avx512f,avx2,fma","avx","sse4.2","default")))
#elif defined(__aarch64__)
#define HYP_TARGET_CLONES \
	__attribute__((target_clones("sve","default")))
#else
#define HYP_TARGET_CLONES
#endif

/* Dimension check (pgvector's CheckDims is not in public header) */
static inline void
HypCheckDims(Vector *a, Vector *b)
{
	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", a->dim, b->dim)));
}

/* Stable arcosh: three-region implementation */
static inline double
stable_arcosh(double x)
{
	double d;

	if (x <= 1.0)
		return 0.0;
	d = x - 1.0;
	if (d < ARCOSH_TAYLOR_THR)
		return sqrt(2.0 * d) * (1.0 - d / 12.0);
	return log(x + sqrt(x * x - 1.0));
}

/* Project vector to Poincaré ball interior (direction-preserving) */
static inline void
project_to_ball(float *x, int dim)
{
	double norm_sq = 0.0;
	double max_norm, scale;
	int i;

	for (i = 0; i < dim; i++)
	{
		double xi = (double) x[i];
		norm_sq += xi * xi;
	}

	max_norm = 1.0 - POINC_EPS;
	if (norm_sq > max_norm * max_norm)
	{
		scale = max_norm / sqrt(norm_sq);
		for (i = 0; i < dim; i++)
			x[i] = (float)((double) x[i] * scale);
	}
}

/* Allocate a new Vector (InitVector is not exported by pgvector) */
static inline Vector *
HypInitVector(int dim)
{
	Vector *result = (Vector *) palloc0(VECTOR_SIZE(dim));

	SET_VARSIZE(result, VECTOR_SIZE(dim));
	result->dim = dim;
	result->unused = 0;
	return result;
}

/* Compute squared norm of vector in double precision */
static inline double
hyp_norm_sq(float *x, int dim)
{
	double s = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double xi = (double) x[i];
		s += xi * xi;
	}
	return s;
}

/* Compute dot product in double precision */
static inline double
hyp_dot(float *x, float *y, int dim)
{
	double s = 0.0;
	int i;

	for (i = 0; i < dim; i++)
		s += (double) x[i] * (double) y[i];
	return s;
}

/* Conformal factor lambda_x = 2 / (1 - ||x||^2) */
static inline double
hyp_lambda(float *x, int dim)
{
	double nsq = hyp_norm_sq(x, dim);

	if (nsq >= 1.0 - POINC_EPS)
		nsq = 1.0 - POINC_EPS;
	return 2.0 / (1.0 - nsq);
}

/* Poincaré ball distance (shared by hyperbolic.c and geometric.c) */
static inline double
PoincareBallDistance(int dim, float *ax, float *bx)
{
	double x_sq = 0.0, y_sq = 0.0, diff_sq = 0.0;
	double denom, arg;
	int i;

	for (i = 0; i < dim; i++)
	{
		double xi = (double) ax[i];
		double yi = (double) bx[i];
		double d  = xi - yi;
		x_sq    += xi * xi;
		y_sq    += yi * yi;
		diff_sq += d * d;
	}

	if (x_sq >= 1.0 - POINC_EPS)
		x_sq = 1.0 - POINC_EPS;
	if (y_sq >= 1.0 - POINC_EPS)
		y_sq = 1.0 - POINC_EPS;

	denom = (1.0 - x_sq) * (1.0 - y_sq);
	arg   = 1.0 + 2.0 * diff_sq / denom;

	return stable_arcosh(arg);
}

#endif /* PG_HYPERBOLIC_H */
