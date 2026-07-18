/*
 * hyp_sqlite.c — Hyperbolic distance functions for SQLite.
 *
 * Loadable extension providing Poincaré ball operations on BLOB vectors.
 * Vectors are stored as packed float32 arrays (little-endian BLOBs).
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o hyp_sqlite.so hyp_sqlite.c -lm
 *
 * Usage:
 *   .load ./hyp_sqlite
 *   SELECT poincare_distance(vec_a, vec_b);
 *   SELECT * FROM vectors ORDER BY poincare_distance(embedding, ?1) LIMIT 10;
 */

#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Stability constants (same as PostgreSQL version) */
#define POINC_EPS         1e-7
#define ARCOSH_TAYLOR_THR 1e-8

/* ---- Shared math (identical to pg_hyperbolic) ---- */

static double
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

static double
norm_sq(const float *x, int dim)
{
	double s = 0.0;
	int i;

	for (i = 0; i < dim; i++)
	{
		double xi = (double)x[i];
		s += xi * xi;
	}
	return s;
}

static double
dot_prod(const float *x, const float *y, int dim)
{
	double s = 0.0;
	int i;

	for (i = 0; i < dim; i++)
		s += (double)x[i] * (double)y[i];
	return s;
}

static double
poincare_dist(const float *ax, const float *bx, int dim, double c)
{
	double x_sq = 0.0, y_sq = 0.0, diff_sq = 0.0;
	double dx, dy, arg;
	int i;

	for (i = 0; i < dim; i++)
	{
		double xi = (double)ax[i];
		double yi = (double)bx[i];
		double d = xi - yi;
		x_sq += xi * xi;
		y_sq += yi * yi;
		diff_sq += d * d;
	}

	dx = 1.0 - c * x_sq;
	dy = 1.0 - c * y_sq;
	if (dx < POINC_EPS) dx = POINC_EPS;
	if (dy < POINC_EPS) dy = POINC_EPS;

	arg = 1.0 + 2.0 * c * diff_sq / (dx * dy);
	return stable_arcosh(arg) / sqrt(c);
}

/* ---- Helper: extract float vector from BLOB ---- */

static const float *
blob_to_vec(sqlite3_value *val, int *dim_out)
{
	int nbytes = sqlite3_value_bytes(val);

	if (nbytes < 4 || nbytes % 4 != 0)
		return NULL;
	*dim_out = nbytes / 4;
	return (const float *)sqlite3_value_blob(val);
}

/* ---- SQLite functions ---- */

/* poincare_distance(blob_a, blob_b [, curvature]) → float */
static void
fn_poincare_distance(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	const float *a, *b;
	int dim_a, dim_b;
	double c = 1.0;

	a = blob_to_vec(argv[0], &dim_a);
	b = blob_to_vec(argv[1], &dim_b);

	if (!a || !b)
	{
		sqlite3_result_error(ctx, "arguments must be float32 BLOBs", -1);
		return;
	}
	if (dim_a != dim_b)
	{
		sqlite3_result_error(ctx, "vectors must have same dimension", -1);
		return;
	}
	if (argc >= 3)
	{
		c = sqlite3_value_double(argv[2]);
		if (c <= 0.0)
		{
			sqlite3_result_error(ctx, "curvature must be positive", -1);
			return;
		}
	}

	sqlite3_result_double(ctx, poincare_dist(a, b, dim_a, c));
}

/* hyp_radius(blob [, curvature]) → float */
static void
fn_hyp_radius(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	const float *v;
	int dim;
	double c = 1.0, norm, sc_norm;

	v = blob_to_vec(argv[0], &dim);
	if (!v)
	{
		sqlite3_result_error(ctx, "argument must be a float32 BLOB", -1);
		return;
	}
	if (argc >= 2)
		c = sqlite3_value_double(argv[1]);

	norm = sqrt(norm_sq(v, dim));
	sc_norm = sqrt(c) * norm;
	if (sc_norm >= 1.0 - POINC_EPS)
		sc_norm = 1.0 - POINC_EPS;

	sqlite3_result_double(ctx, (2.0 / sqrt(c)) * atanh(sc_norm));
}

/* hyp_check_ball(blob [, curvature]) → 0 or 1 */
static void
fn_hyp_check_ball(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	const float *v;
	int dim;
	double c = 1.0, nsq;

	v = blob_to_vec(argv[0], &dim);
	if (!v)
	{
		sqlite3_result_error(ctx, "argument must be a float32 BLOB", -1);
		return;
	}
	if (argc >= 2)
		c = sqlite3_value_double(argv[1]);

	nsq = norm_sq(v, dim);
	sqlite3_result_int(ctx, nsq < (1.0 / c));
}

/* hyp_mobius_add(blob_a, blob_b [, curvature]) → blob */
static void
fn_hyp_mobius_add(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	const float *a, *b;
	int dim_a, dim_b, i;
	double c = 1.0;
	double x_sq, y_sq, xy, denom, c1, c2;
	float *result;

	a = blob_to_vec(argv[0], &dim_a);
	b = blob_to_vec(argv[1], &dim_b);

	if (!a || !b || dim_a != dim_b)
	{
		sqlite3_result_error(ctx, "invalid vector arguments", -1);
		return;
	}
	if (argc >= 3)
		c = sqlite3_value_double(argv[2]);

	x_sq = norm_sq(a, dim_a);
	y_sq = norm_sq(b, dim_a);
	xy = dot_prod(a, b, dim_a);

	denom = 1.0 + 2.0 * c * xy + c * c * x_sq * y_sq;
	if (fabs(denom) < 1e-15) denom = 1e-15;

	c1 = 1.0 + 2.0 * c * xy + c * y_sq;
	c2 = 1.0 - c * x_sq;

	result = (float *)sqlite3_malloc(dim_a * sizeof(float));
	for (i = 0; i < dim_a; i++)
		result[i] = (float)((c1 * (double)a[i] + c2 * (double)b[i]) / denom);

	sqlite3_result_blob(ctx, result, dim_a * sizeof(float), sqlite3_free);
}

/* hyp_midpoint(blob_a, blob_b) → blob */
static void
fn_hyp_midpoint(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	const float *a, *b;
	int dim_a, dim_b, i;
	double la, lb, la2, lb2, denom, nsq_a, nsq_b;
	float *result;

	a = blob_to_vec(argv[0], &dim_a);
	b = blob_to_vec(argv[1], &dim_b);

	if (!a || !b || dim_a != dim_b)
	{
		sqlite3_result_error(ctx, "invalid vector arguments", -1);
		return;
	}

	nsq_a = norm_sq(a, dim_a);
	nsq_b = norm_sq(b, dim_a);
	if (nsq_a >= 1.0 - POINC_EPS) nsq_a = 1.0 - POINC_EPS;
	if (nsq_b >= 1.0 - POINC_EPS) nsq_b = 1.0 - POINC_EPS;

	la = 2.0 / (1.0 - nsq_a);
	lb = 2.0 / (1.0 - nsq_b);
	la2 = la * la;
	lb2 = lb * lb;
	denom = la2 + lb2 - 2.0;
	if (fabs(denom) < 1e-15) denom = 1e-15;

	result = (float *)sqlite3_malloc(dim_a * sizeof(float));
	for (i = 0; i < dim_a; i++)
		result[i] = (float)((lb2 * (double)a[i] + la2 * (double)b[i]) / denom);

	sqlite3_result_blob(ctx, result, dim_a * sizeof(float), sqlite3_free);
}

/* poincare_distance_lorentz(blob_a, blob_b) → float
 * Alternative distance via Lorentz model conversion.
 * Same result as poincare_distance, different computation path.
 */
static void
fn_poincare_distance_lorentz(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	const float *a, *b;
	int dim_a, dim_b, i;
	double a_sq = 0.0, b_sq = 0.0, ab = 0.0;
	double a_denom, b_denom, a0, b0, spatial_ip, lorentz_ip;

	a = blob_to_vec(argv[0], &dim_a);
	b = blob_to_vec(argv[1], &dim_b);

	if (!a || !b)
	{
		sqlite3_result_error(ctx, "arguments must be float32 BLOBs", -1);
		return;
	}
	if (dim_a != dim_b)
	{
		sqlite3_result_error(ctx, "vectors must have same dimension", -1);
		return;
	}

	for (i = 0; i < dim_a; i++)
	{
		double ai = (double)a[i];
		double bi = (double)b[i];
		a_sq += ai * ai;
		b_sq += bi * bi;
		ab   += ai * bi;
	}

	if (a_sq >= 1.0 - POINC_EPS) a_sq = 1.0 - POINC_EPS;
	if (b_sq >= 1.0 - POINC_EPS) b_sq = 1.0 - POINC_EPS;

	a_denom = 1.0 - a_sq;
	b_denom = 1.0 - b_sq;

	a0 = (1.0 + a_sq) / a_denom;
	b0 = (1.0 + b_sq) / b_denom;
	spatial_ip = 4.0 * ab / (a_denom * b_denom);
	lorentz_ip = -a0 * b0 + spatial_ip;

	sqlite3_result_double(ctx, stable_arcosh(-lorentz_ip));
}

/* vec_from_floats(f1, f2, ...) → blob — helper to create vectors */
static void
fn_vec_from_floats(sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
	float *buf;
	int i;

	buf = (float *)sqlite3_malloc(argc * sizeof(float));
	for (i = 0; i < argc; i++)
		buf[i] = (float)sqlite3_value_double(argv[i]);

	sqlite3_result_blob(ctx, buf, argc * sizeof(float), sqlite3_free);
}

/* ---- Extension entry point ---- */

#ifdef _WIN32
__declspec(dllexport)
#endif
int
sqlite3_hypsqlite_init(sqlite3 *db, char **err, const sqlite3_api_routines *api)
{
	SQLITE_EXTENSION_INIT2(api);

	sqlite3_create_function(db, "poincare_distance", 2, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_poincare_distance, 0, 0);
	sqlite3_create_function(db, "poincare_distance", 3, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_poincare_distance, 0, 0);

	sqlite3_create_function(db, "hyp_radius", 1, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_radius, 0, 0);
	sqlite3_create_function(db, "hyp_radius", 2, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_radius, 0, 0);

	sqlite3_create_function(db, "hyp_check_ball", 1, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_check_ball, 0, 0);
	sqlite3_create_function(db, "hyp_check_ball", 2, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_check_ball, 0, 0);

	sqlite3_create_function(db, "hyp_mobius_add", 2, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_mobius_add, 0, 0);
	sqlite3_create_function(db, "hyp_mobius_add", 3, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_mobius_add, 0, 0);

	sqlite3_create_function(db, "hyp_midpoint", 2, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_hyp_midpoint, 0, 0);

	sqlite3_create_function(db, "poincare_distance_lorentz", 2, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_poincare_distance_lorentz, 0, 0);

	sqlite3_create_function(db, "vec_from_floats", -1, SQLITE_UTF8|SQLITE_DETERMINISTIC, 0,
							fn_vec_from_floats, 0, 0);

	return SQLITE_OK;
}
