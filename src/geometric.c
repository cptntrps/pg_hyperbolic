#include "hyperbolic.h"
#include "funcapi.h"

/*
 * Internal: Möbius addition x ⊕ y in the Poincaré ball (c=1).
 * Formula: ((1+2<x,y>+||y||²)x + (1-||x||²)y) / (1+2<x,y>+||x||²||y||²)
 * Result written into out[] (must be pre-allocated, length dim).
 */
static void
mobius_add(float *x, float *y, int dim, double *out)
{
	double x_sq, y_sq, xy, denom, c1, c2;
	int i;

	x_sq = hyp_norm_sq(x, dim);
	y_sq = hyp_norm_sq(y, dim);
	xy   = hyp_dot(x, y, dim);

	if (x_sq >= 1.0 - POINC_EPS) x_sq = 1.0 - POINC_EPS;
	if (y_sq >= 1.0 - POINC_EPS) y_sq = 1.0 - POINC_EPS;

	denom = 1.0 + 2.0 * xy + x_sq * y_sq;
	if (fabs(denom) < 1e-15) denom = 1e-15;

	c1 = 1.0 + 2.0 * xy + y_sq;
	c2 = 1.0 - x_sq;

	for (i = 0; i < dim; i++)
		out[i] = (c1 * (double) x[i] + c2 * (double) y[i]) / denom;
}

/*
 * Internal: negate a float vector into double buffer.
 */
static void
negate_vec(float *x, int dim, float *out)
{
	int i;

	for (i = 0; i < dim; i++)
		out[i] = -x[i];
}

/*
 * Internal: write double buffer to Vector result, projecting to ball.
 */
static Vector *
doubles_to_vector(double *buf, int dim)
{
	Vector *result = HypInitVector(dim);
	int i;

	for (i = 0; i < dim; i++)
		result->x[i] = (float) buf[i];
	project_to_ball(result->x, dim);
	return result;
}

/* ============================================================
 * hyp_mobius_add(vector, vector) → vector
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_mobius_add);
Datum
hyp_mobius_add(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	double *buf;

	HypCheckDims(a, b);
	buf = (double *) palloc(sizeof(double) * a->dim);
	mobius_add(a->x, b->x, a->dim, buf);
	PG_RETURN_VECTOR_P(doubles_to_vector(buf, a->dim));
}

/* ============================================================
 * hyp_midpoint(vector, vector) → vector
 * Conformal factor midpoint: m = (λ²_b·a + λ²_a·b) / (λ²_b + λ²_a - 2)
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_midpoint);
Datum
hyp_midpoint(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	double la, lb, la2, lb2, denom;
	double *buf;
	int i;

	HypCheckDims(a, b);

	la = hyp_lambda(a->x, a->dim);
	lb = hyp_lambda(b->x, b->dim);
	la2 = la * la;
	lb2 = lb * lb;
	denom = la2 + lb2 - 2.0;
	if (fabs(denom) < 1e-15) denom = 1e-15;

	buf = (double *) palloc(sizeof(double) * a->dim);
	for (i = 0; i < a->dim; i++)
		buf[i] = (lb2 * (double) a->x[i] + la2 * (double) b->x[i]) / denom;

	PG_RETURN_VECTOR_P(doubles_to_vector(buf, a->dim));
}

/* ============================================================
 * hyp_exp_map(base, tangent) → vector
 * exp_x(v) = x ⊕ (tanh(λ_x·||v||/2) · v/||v||)
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_exp_map);
Datum
hyp_exp_map(PG_FUNCTION_ARGS)
{
	Vector *base = PG_GETARG_VECTOR_P(0);
	Vector *tang = PG_GETARG_VECTOR_P(1);
	double lam, v_norm, scale;
	float *scaled;
	double *buf;
	int i;

	HypCheckDims(base, tang);

	lam = hyp_lambda(base->x, base->dim);
	v_norm = sqrt(hyp_norm_sq(tang->x, tang->dim));

	if (v_norm < 1e-15)
	{
		/* Zero tangent → return base point */
		Vector *result = HypInitVector(base->dim);
		memcpy(result->x, base->x, sizeof(float) * base->dim);
		PG_RETURN_VECTOR_P(result);
	}

	scale = tanh(lam * v_norm / 2.0) / v_norm;
	scaled = (float *) palloc(sizeof(float) * tang->dim);
	for (i = 0; i < tang->dim; i++)
		scaled[i] = (float)(scale * (double) tang->x[i]);

	buf = (double *) palloc(sizeof(double) * base->dim);
	mobius_add(base->x, scaled, base->dim, buf);
	pfree(scaled);
	PG_RETURN_VECTOR_P(doubles_to_vector(buf, base->dim));
}

/* ============================================================
 * hyp_log_map(base, point) → vector (tangent at base)
 * log_x(y) = (2/(λ_x·α)) · atanh(α) · u, where u=(-x)⊕y, α=||u||
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_log_map);
Datum
hyp_log_map(PG_FUNCTION_ARGS)
{
	Vector *base = PG_GETARG_VECTOR_P(0);
	Vector *point = PG_GETARG_VECTOR_P(1);
	double lam, alpha, coeff;
	float *neg_base;
	double *u;
	Vector *result;
	int i;

	HypCheckDims(base, point);

	neg_base = (float *) palloc(sizeof(float) * base->dim);
	negate_vec(base->x, base->dim, neg_base);

	u = (double *) palloc(sizeof(double) * base->dim);
	mobius_add(neg_base, point->x, base->dim, u);
	pfree(neg_base);

	alpha = 0.0;
	for (i = 0; i < base->dim; i++)
		alpha += u[i] * u[i];
	alpha = sqrt(alpha);

	if (alpha < 1e-15)
	{
		/* Points are the same → zero tangent */
		result = HypInitVector(base->dim);
		pfree(u);
		PG_RETURN_VECTOR_P(result);
	}

	if (alpha >= 1.0 - POINC_EPS)
		alpha = 1.0 - POINC_EPS;

	lam = hyp_lambda(base->x, base->dim);
	coeff = (2.0 / (lam * alpha)) * atanh(alpha);

	result = HypInitVector(base->dim);
	for (i = 0; i < base->dim; i++)
		result->x[i] = (float)(coeff * u[i]);

	pfree(u);
	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_lerp(a, b, t) → vector
 * Geodesic interpolation: exp_a(t · log_a(b))
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_lerp);
Datum
hyp_lerp(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	double t = PG_GETARG_FLOAT8(2);
	double lam_a, alpha, coeff, v_norm, scale;
	float *neg_a, *scaled;
	double *u, *tangent, *buf;
	int i;

	HypCheckDims(a, b);

	/* log_a(b) */
	neg_a = (float *) palloc(sizeof(float) * a->dim);
	negate_vec(a->x, a->dim, neg_a);
	u = (double *) palloc(sizeof(double) * a->dim);
	mobius_add(neg_a, b->x, a->dim, u);
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
	if (alpha >= 1.0 - POINC_EPS)
		alpha = 1.0 - POINC_EPS;

	lam_a = hyp_lambda(a->x, a->dim);
	coeff = (2.0 / (lam_a * alpha)) * atanh(alpha);

	/* tangent = t * log_a(b) */
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

	/* exp_a(t * tangent) */
	scale = tanh(lam_a * v_norm / 2.0) / v_norm;
	scaled = (float *) palloc(sizeof(float) * a->dim);
	for (i = 0; i < a->dim; i++)
		scaled[i] = (float)(scale * tangent[i]);
	pfree(tangent);

	buf = (double *) palloc(sizeof(double) * a->dim);
	mobius_add(a->x, scaled, a->dim, buf);
	pfree(scaled);

	PG_RETURN_VECTOR_P(doubles_to_vector(buf, a->dim));
}

/* ============================================================
 * hyp_cone_contains(parent, child, aperture_K) → boolean
 * True if child is in parent's entailment cone.
 * Half-aperture: ψ(x) = arcsin(K·(1-||x||²)/||x||)
 * Test: angle between direction to child and parent's direction < ψ
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_cone_contains);
Datum
hyp_cone_contains(PG_FUNCTION_ARGS)
{
	Vector *parent = PG_GETARG_VECTOR_P(0);
	Vector *child  = PG_GETARG_VECTOR_P(1);
	double K = PG_GETARG_FLOAT8(2);
	double p_norm, p_sq, half_aperture, cos_angle;
	double *u;
	float *neg_p;
	double u_norm, p_dot_u;
	int i;

	HypCheckDims(parent, child);

	p_sq = hyp_norm_sq(parent->x, parent->dim);
	p_norm = sqrt(p_sq);
	if (p_norm < 1e-15)
		PG_RETURN_BOOL(true); /* origin contains everything */

	if (p_sq >= 1.0 - POINC_EPS)
		p_sq = 1.0 - POINC_EPS;

	/* Half-aperture of parent's cone */
	half_aperture = asin(fmin(1.0, K * (1.0 - p_sq) / p_norm));

	/* Direction from parent to child: (-p) ⊕ c */
	neg_p = (float *) palloc(sizeof(float) * parent->dim);
	negate_vec(parent->x, parent->dim, neg_p);
	u = (double *) palloc(sizeof(double) * parent->dim);
	mobius_add(neg_p, child->x, parent->dim, u);
	pfree(neg_p);

	u_norm = 0.0;
	for (i = 0; i < parent->dim; i++)
		u_norm += u[i] * u[i];
	u_norm = sqrt(u_norm);

	if (u_norm < 1e-15)
	{
		pfree(u);
		PG_RETURN_BOOL(true); /* same point */
	}

	/* Angle between parent's direction and the direction to child */
	p_dot_u = 0.0;
	for (i = 0; i < parent->dim; i++)
		p_dot_u += (double) parent->x[i] * u[i];
	pfree(u);

	cos_angle = p_dot_u / (p_norm * u_norm);
	if (cos_angle > 1.0) cos_angle = 1.0;
	if (cos_angle < -1.0) cos_angle = -1.0;

	PG_RETURN_BOOL(acos(cos_angle) <= half_aperture);
}

/* ============================================================
 * hyp_busemann(point, direction) → float8
 * B_θ(x) = ln((1 - ||x||²) / ||x - θ||²)
 * θ is normalized to unit vector internally.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_busemann);
Datum
hyp_busemann(PG_FUNCTION_ARGS)
{
	Vector *point = PG_GETARG_VECTOR_P(0);
	Vector *dir   = PG_GETARG_VECTOR_P(1);
	double x_sq, dir_norm, diff_sq;
	int i;

	HypCheckDims(point, dir);

	x_sq = hyp_norm_sq(point->x, point->dim);
	if (x_sq >= 1.0 - POINC_EPS) x_sq = 1.0 - POINC_EPS;

	dir_norm = sqrt(hyp_norm_sq(dir->x, dir->dim));
	if (dir_norm < 1e-15)
		PG_RETURN_FLOAT8(0.0);

	/* ||x - θ_hat||² where θ_hat = dir/||dir|| */
	diff_sq = 0.0;
	for (i = 0; i < point->dim; i++)
	{
		double d = (double) point->x[i] - (double) dir->x[i] / dir_norm;
		diff_sq += d * d;
	}
	if (diff_sq < 1e-15) diff_sq = 1e-15;

	PG_RETURN_FLOAT8(log((1.0 - x_sq) / diff_sq));
}

/* ============================================================
 * hyp_parallel_transport(tangent, from, to) → vector
 * Transport tangent v from T_from to T_to.
 * Uses: PT = (λ_from / λ_to) · gyration[to, -from](v)
 * Simplified: PT_{0→y}(v) = v · (1 - ||y||²) / 2
 * General case via log/exp composition.
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_parallel_transport);
Datum
hyp_parallel_transport(PG_FUNCTION_ARGS)
{
	Vector *tang = PG_GETARG_VECTOR_P(0);
	Vector *from = PG_GETARG_VECTOR_P(1);
	Vector *to   = PG_GETARG_VECTOR_P(2);
	double lam_from, lam_to, ratio;
	float *neg_from;
	double *u, *gyrated;
	Vector *result;
	int i;

	if (from->dim != to->dim || from->dim != tang->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("all vectors must have same dimensions")));

	lam_from = hyp_lambda(from->x, from->dim);
	lam_to   = hyp_lambda(to->x, to->dim);
	ratio = lam_from / lam_to;

	/* Gyration: gyr[to, -from](v) ≈ (-from ⊕ to ⊕ v) simplified */
	/* Use the formula: gyr[a,b](v) = -(a⊕b) ⊕ (a ⊕ (b⊕v)) */
	neg_from = (float *) palloc(sizeof(float) * from->dim);
	negate_vec(from->x, from->dim, neg_from);

	/* Step 1: b⊕v = neg_from ⊕ tang */
	u = (double *) palloc(sizeof(double) * from->dim);
	mobius_add(neg_from, tang->x, from->dim, u);

	/* Step 2: a ⊕ (b⊕v) = to ⊕ u -- need float* for u */
	{
		float *u_f = (float *) palloc(sizeof(float) * from->dim);
		double *step2 = (double *) palloc(sizeof(double) * from->dim);
		double *ab = (double *) palloc(sizeof(double) * from->dim);
		float *ab_f, *neg_ab_f;

		for (i = 0; i < from->dim; i++)
			u_f[i] = (float) u[i];
		mobius_add(to->x, u_f, from->dim, step2);
		pfree(u_f);

		/* Step 3: a⊕b = to ⊕ neg_from */
		mobius_add(to->x, neg_from, from->dim, ab);

		/* Step 4: -(a⊕b) ⊕ step2 */
		ab_f = (float *) palloc(sizeof(float) * from->dim);
		neg_ab_f = (float *) palloc(sizeof(float) * from->dim);
		for (i = 0; i < from->dim; i++)
		{
			ab_f[i] = (float) ab[i];
			neg_ab_f[i] = (float)(-ab[i]);
		}

		gyrated = (double *) palloc(sizeof(double) * from->dim);
		{
			float *s2_f = (float *) palloc(sizeof(float) * from->dim);
			for (i = 0; i < from->dim; i++)
				s2_f[i] = (float) step2[i];
			mobius_add(neg_ab_f, s2_f, from->dim, gyrated);
			pfree(s2_f);
		}

		pfree(ab_f);
		pfree(neg_ab_f);
		pfree(ab);
		pfree(step2);
	}
	pfree(neg_from);
	pfree(u);

	/* Scale by λ_from / λ_to */
	result = HypInitVector(from->dim);
	for (i = 0; i < from->dim; i++)
		result->x[i] = (float)(ratio * gyrated[i]);
	pfree(gyrated);

	PG_RETURN_VECTOR_P(result);
}

/* ============================================================
 * hyp_analogy(a, b, c) → vector
 * "a is to b as c is to ?" = exp_c(PT_{a→c}(log_a(b)))
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_analogy);
Datum
hyp_analogy(PG_FUNCTION_ARGS)
{
	Vector *a = PG_GETARG_VECTOR_P(0);
	Vector *b = PG_GETARG_VECTOR_P(1);
	Vector *c = PG_GETARG_VECTOR_P(2);
	Datum log_ab, transported, result;

	if (a->dim != b->dim || a->dim != c->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("all vectors must have same dimensions")));

	/* log_a(b) */
	log_ab = DirectFunctionCall2(hyp_log_map,
								PG_GETARG_DATUM(0), PG_GETARG_DATUM(1));

	/* PT_{a→c}(log_a(b)) */
	transported = DirectFunctionCall3(hyp_parallel_transport,
									 log_ab, PG_GETARG_DATUM(0),
									 PG_GETARG_DATUM(2));

	/* exp_c(transported) */
	result = DirectFunctionCall2(hyp_exp_map,
								PG_GETARG_DATUM(2), transported);

	PG_RETURN_DATUM(result);
}

/* ============================================================
 * hyp_dist_to_geodesic(point, geo_a, geo_b) → float8
 * Distance from point to the geodesic defined by geo_a and geo_b.
 * Uses hyperbolic law of sines: d = arcsinh(|sinh(d(a,p))·sin(∠a)|)
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_dist_to_geodesic);
Datum
hyp_dist_to_geodesic(PG_FUNCTION_ARGS)
{
	Vector *p = PG_GETARG_VECTOR_P(0);
	Vector *ga = PG_GETARG_VECTOR_P(1);
	Vector *gb = PG_GETARG_VECTOR_P(2);
	double d_ap, d_ab, d_bp, cos_A, sin_A, result;

	if (p->dim != ga->dim || p->dim != gb->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("all vectors must have same dimensions")));

	/* Compute pairwise distances */
	d_ap = PoincareBallDistance(ga->dim, ga->x, p->x);
	d_ab = PoincareBallDistance(ga->dim, ga->x, gb->x);
	d_bp = PoincareBallDistance(gb->dim, gb->x, p->x);

	if (d_ab < 1e-15 || d_ap < 1e-15)
		PG_RETURN_FLOAT8(0.0);

	/* Hyperbolic law of cosines: cosh(d_bp) = cosh(d_ap)·cosh(d_ab) - sinh(d_ap)·sinh(d_ab)·cos(A) */
	cos_A = (cosh(d_ap) * cosh(d_ab) - cosh(d_bp)) / (sinh(d_ap) * sinh(d_ab));
	if (cos_A > 1.0) cos_A = 1.0;
	if (cos_A < -1.0) cos_A = -1.0;
	sin_A = sqrt(1.0 - cos_A * cos_A);

	/* Distance = arcsinh(|sinh(d_ap) · sin(A)|) */
	result = asinh(fabs(sinh(d_ap) * sin_A));

	PG_RETURN_FLOAT8(result);
}

/* ============================================================
 * hyp_ideal_direction(vector) → vector
 * Normalize to unit sphere: x / ||x||
 * ============================================================ */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hyp_ideal_direction);
Datum
hyp_ideal_direction(PG_FUNCTION_ARGS)
{
	Vector *v = PG_GETARG_VECTOR_P(0);
	double norm;
	Vector *result;
	int i;

	norm = sqrt(hyp_norm_sq(v->x, v->dim));
	result = HypInitVector(v->dim);

	if (norm < 1e-15)
	{
		PG_RETURN_VECTOR_P(result); /* zero vector stays zero */
	}

	for (i = 0; i < v->dim; i++)
		result->x[i] = (float)((double) v->x[i] / norm);

	PG_RETURN_VECTOR_P(result);
}
