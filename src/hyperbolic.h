#ifndef PG_HYPERBOLIC_H
#define PG_HYPERBOLIC_H

#include "postgres.h"
#include "fmgr.h"
#include "extension/vector/vector.h"
#include <math.h>

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
	for (int i = 0; i < dim; i++)
	{
		double xi = (double) x[i];
		norm_sq += xi * xi;
	}
	double max_norm;
	double scale;

	max_norm = 1.0 - POINC_EPS;
	if (norm_sq > max_norm * max_norm)
	{
		scale = max_norm / sqrt(norm_sq);
		for (int i = 0; i < dim; i++)
			x[i] = (float)((double) x[i] * scale);
	}
}

#endif /* PG_HYPERBOLIC_H */
