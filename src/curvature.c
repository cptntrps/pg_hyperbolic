#include "hyperbolic.h"

/*
 * Variable-curvature Poincaré ball operations.
 *
 * All functions take curvature c > 0 as the last parameter.
 * Curvature c means sectional curvature = -c, ball radius = 1/√c.
 * For c = 1.0, results are identical to the standard (c=1) functions.
 */

static Vector *
doubles_to_vector_c(double *buf, int dim, double c)
{
	Vector *result = HypInitVector(dim);
	int i;

	for (i = 0; i < dim; i++)
		result->x[i] = (float) buf[i];
	project_to_ball_c(result->x, dim, c);
	return result;
}

/* ============================================================
 * poincare_distance_c(vector, vector, float8) → float8
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(poincare_distance_c);
Datum
poincare_distance_c(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	double  c = PG_GETARG_FLOAT8(2);

	HypCheckDims(a, b);

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	PG_RETURN_FLOAT8(PoincareBallDistance_c(a->dim, a->x, b->x, c));
}

/* ============================================================
 * hyp_mobius_add_c(vector, vector, float8) → vector
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_mobius_add_c);
Datum
hyp_mobius_add_c(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	double  c = PG_GETARG_FLOAT8(2);
	double *buf;

	HypCheckDims(a, b);

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	buf = (double *) palloc(sizeof(double) * a->dim);
	mobius_add_c(a->x, b->x, a->dim, c, buf);
	PG_RETURN_VECTOR_P(doubles_to_vector_c(buf, a->dim, c));
}

/* ============================================================
 * hyp_radius_c(vector, float8) → float8
 * d_c(0, x) = (2/√c) · atanh(√c · ||x||)
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_radius_c);
Datum
hyp_radius_c(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double  c = PG_GETARG_FLOAT8(1);
	double  norm, sc_norm, max_arg;

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	norm = sqrt(hyp_norm_sq(v->x, v->dim));
	sc_norm = sqrt(c) * norm;
	max_arg = 1.0 - POINC_EPS;
	if (sc_norm >= max_arg)
		sc_norm = max_arg;

	PG_RETURN_FLOAT8((2.0 / sqrt(c)) * atanh(sc_norm));
}

/* ============================================================
 * hyp_exp_map_c(base, tangent, curvature) → vector
 * exp_x^c(v) = x ⊕_c (tanh(√c·λ_x^c·||v||/2) · v / (√c·||v||))
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_exp_map_c);
Datum
hyp_exp_map_c(PG_FUNCTION_ARGS)
{
	Vector *base = PG_GETARG_VECTOR_P(0);
	Vector *tang = PG_GETARG_VECTOR_P(1);
	double  c = PG_GETARG_FLOAT8(2);
	double  sc, lam, v_norm, scale;
	float  *scaled;
	double *buf;
	int i;

	HypCheckDims(base, tang);

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	sc = sqrt(c);
	lam = hyp_lambda_c(base->x, base->dim, c);
	v_norm = sqrt(hyp_norm_sq(tang->x, tang->dim));

	if (v_norm < 1e-15)
	{
		Vector *result = HypInitVector(base->dim);
		memcpy(result->x, base->x, sizeof(float) * base->dim);
		PG_RETURN_VECTOR_P(result);
	}

	scale = tanh(sc * lam * v_norm / 2.0) / (sc * v_norm);
	scaled = (float *) palloc(sizeof(float) * tang->dim);
	for (i = 0; i < tang->dim; i++)
		scaled[i] = (float)(scale * (double) tang->x[i]);

	buf = (double *) palloc(sizeof(double) * base->dim);
	mobius_add_c(base->x, scaled, base->dim, c, buf);
	pfree(scaled);
	PG_RETURN_VECTOR_P(doubles_to_vector_c(buf, base->dim, c));
}

/* ============================================================
 * hyp_log_map_c(base, point, curvature) → vector
 * log_x^c(y) = (2/(√c·λ_x^c)) · atanh(√c·||u||) · u/||u||
 *   where u = (-x) ⊕_c y
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_log_map_c);
Datum
hyp_log_map_c(PG_FUNCTION_ARGS)
{
	Vector *base = PG_GETARG_VECTOR_P(0);
	Vector *point = PG_GETARG_VECTOR_P(1);
	double  c = PG_GETARG_FLOAT8(2);
	double  sc, lam, alpha, sc_alpha, coeff;
	float  *neg_base;
	double *u;
	Vector *result;
	int i;

	HypCheckDims(base, point);

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	sc = sqrt(c);
	neg_base = (float *) palloc(sizeof(float) * base->dim);
	for (i = 0; i < base->dim; i++)
		neg_base[i] = -base->x[i];

	u = (double *) palloc(sizeof(double) * base->dim);
	mobius_add_c(neg_base, point->x, base->dim, c, u);
	pfree(neg_base);

	alpha = 0.0;
	for (i = 0; i < base->dim; i++)
		alpha += u[i] * u[i];
	alpha = sqrt(alpha);

	if (alpha < 1e-15)
	{
		result = HypInitVector(base->dim);
		pfree(u);
		PG_RETURN_VECTOR_P(result);
	}

	sc_alpha = sc * alpha;
	if (sc_alpha >= 1.0 - POINC_EPS)
		sc_alpha = 1.0 - POINC_EPS;

	lam = hyp_lambda_c(base->x, base->dim, c);
	coeff = (2.0 / (sc * lam * alpha)) * atanh(sc_alpha);

	result = HypInitVector(base->dim);
	for (i = 0; i < base->dim; i++)
		result->x[i] = (float)(coeff * u[i]);

	pfree(u);
	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_project_c(vector, float8) → vector
 * Project to ball of radius 1/√c
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_project_c);
Datum
hyp_project_c(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double  c = PG_GETARG_FLOAT8(1);
	Vector *result;

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	result = HypInitVector(v->dim);
	memcpy(result->x, v->x, sizeof(float) * v->dim);
	project_to_ball_c(result->x, v->dim, c);

	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_lerp_c(a, b, t, curvature) → vector
 * Geodesic interpolation with curvature: exp_a^c(t · log_a^c(b))
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_lerp_c);
Datum
hyp_lerp_c(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	double  t = PG_GETARG_FLOAT8(2);
	double  c = PG_GETARG_FLOAT8(3);
	double  sc, lam_a, alpha, sc_alpha, coeff, v_norm, scale;
	float  *neg_a, *scaled;
	double *u, *tangent, *buf;
	int i;

	HypCheckDims(a, b);

	if (c <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("curvature must be positive, got %g", c)));

	sc = sqrt(c);

	/* log_a^c(b) */
	neg_a = (float *) palloc(sizeof(float) * a->dim);
	for (i = 0; i < a->dim; i++)
		neg_a[i] = -a->x[i];
	u = (double *) palloc(sizeof(double) * a->dim);
	mobius_add_c(neg_a, b->x, a->dim, c, u);
	pfree(neg_a);

	alpha = 0.0;
	for (i = 0; i < a->dim; i++)
		alpha += u[i] * u[i];
	alpha = sqrt(alpha);

	if (alpha < 1e-15)
	{
		Vector *result = HypInitVector(a->dim);
		memcpy(result->x, a->x, sizeof(float) * a->dim);
		pfree(u);
		PG_RETURN_VECTOR_P(result);
	}

	sc_alpha = sc * alpha;
	if (sc_alpha >= 1.0 - POINC_EPS)
		sc_alpha = 1.0 - POINC_EPS;

	lam_a = hyp_lambda_c(a->x, a->dim, c);
	coeff = (2.0 / (sc * lam_a * alpha)) * atanh(sc_alpha);

	/* tangent = t * log_a^c(b) */
	tangent = (double *) palloc(sizeof(double) * a->dim);
	v_norm = 0.0;
	for (i = 0; i < a->dim; i++)
	{
		tangent[i] = t * coeff * u[i];
		v_norm += tangent[i] * tangent[i];
	}
	v_norm = sqrt(v_norm);
	pfree(u);

	if (v_norm < 1e-15)
	{
		Vector *result = HypInitVector(a->dim);
		memcpy(result->x, a->x, sizeof(float) * a->dim);
		pfree(tangent);
		PG_RETURN_VECTOR_P(result);
	}

	/* exp_a^c(t * tangent) */
	scale = tanh(sc * lam_a * v_norm / 2.0) / (sc * v_norm);
	scaled = (float *) palloc(sizeof(float) * a->dim);
	for (i = 0; i < a->dim; i++)
		scaled[i] = (float)(scale * tangent[i]);
	pfree(tangent);

	buf = (double *) palloc(sizeof(double) * a->dim);
	mobius_add_c(a->x, scaled, a->dim, c, buf);
	pfree(scaled);

	PG_RETURN_VECTOR_P(doubles_to_vector_c(buf, a->dim, c));
}
