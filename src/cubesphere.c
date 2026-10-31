#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cubesphere.h"

const vec3 cs_face_center[6] = { {0, 0, 1}, {1, 0, 0}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0} };
const vec3 cs_face_right[6]  = { {1, 0, 0}, {0, 0, -1}, {-1, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0} };
const vec3 cs_face_up[6]     = { {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1} };

int cs_face_of(vec3 d)
{
	double ax = fabs(d.x), ay = fabs(d.y), az = fabs(d.z);
	if (ax >= ay && ax >= az)
		return d.x > 0 ? 1 : 3;
	if (ay >= az)
		return d.y > 0 ? 4 : 5;
	return d.z > 0 ? 0 : 2;
}

vec3 cs_face_ab_to_dir(int f, double a, double b)
{
	vec3 d = v3_add(cs_face_center[f],
			v3_add(v3_scale(cs_face_right[f], tan(a)), v3_scale(cs_face_up[f], tan(b))));
	return v3_normalize(d);
}

void cs_locate(const CubeSphere *cs, vec3 d, int *f, double *x, double *y)
{
	int face = cs_face_of(d);
	double dc = v3_dot(d, cs_face_center[face]);
	double u = v3_dot(d, cs_face_right[face]) / dc;
	double v = v3_dot(d, cs_face_up[face]) / dc;
	*f = face;
	*x = (atan(u) + M_PI / 4) / cs->h - 0.5;
	*y = (atan(v) + M_PI / 4) / cs->h - 0.5;
}

int cs_cell_of(const CubeSphere *cs, vec3 d)
{
	int f;
	double x, y;
	cs_locate(cs, d, &f, &x, &y);
	int i = (int) floor(x + 0.5), j = (int) floor(y + 0.5);
	if (i < 0) i = 0;
	if (i >= cs->N) i = cs->N - 1;
	if (j < 0) j = 0;
	if (j >= cs->N) j = cs->N - 1;
	return cs_index(cs, f, i, j);
}

static vec3 cell_dir(const CubeSphere *cs, int f, int i, int j)
{
	return cs_face_ab_to_dir(f, -M_PI / 4 + (i + 0.5) * cs->h, -M_PI / 4 + (j + 0.5) * cs->h);
}

/* Index of cell (i, j) on face f; out-of-range (ghost) cells are resolved by
 * extending face f's coordinates and finding the real cell containing that
 * direction. */
static int tap_index(const CubeSphere *cs, int f, int i, int j)
{
	if (i >= 0 && j >= 0 && i < cs->N && j < cs->N)
		return cs_index(cs, f, i, j);
	return cs_cell_of(cs, cell_dir(cs, f, i, j));
}

void cs_taps(const CubeSphere *cs, vec3 p, int idx[4], double wt[4])
{
	int f;
	double x, y;
	cs_locate(cs, p, &f, &x, &y);
	int i0 = (int) floor(x), j0 = (int) floor(y);
	double fx = x - i0, fy = y - j0;
	int N = cs->N;
	if (i0 >= 0 && j0 >= 0 && i0 + 1 < N && j0 + 1 < N) {
		int b = cs_index(cs, f, i0, j0);
		idx[0] = b;
		idx[1] = b + 1;
		idx[2] = b + N;
		idx[3] = b + N + 1;
	} else {
		idx[0] = tap_index(cs, f, i0, j0);
		idx[1] = tap_index(cs, f, i0 + 1, j0);
		idx[2] = tap_index(cs, f, i0, j0 + 1);
		idx[3] = tap_index(cs, f, i0 + 1, j0 + 1);
	}
	wt[0] = (1 - fx) * (1 - fy);
	wt[1] = fx * (1 - fy);
	wt[2] = (1 - fx) * fy;
	wt[3] = fx * fy;
}

double cs_sample_scalar(const CubeSphere *cs, const float *fld, vec3 p)
{
	int idx[4];
	double wt[4];
	cs_taps(cs, p, idx, wt);
	return wt[0] * fld[idx[0]] + wt[1] * fld[idx[1]] + wt[2] * fld[idx[2]] + wt[3] * fld[idx[3]];
}

vec3 cs_sample_vec(const CubeSphere *cs, const float *fld3, vec3 p)
{
	int idx[4];
	double wt[4];
	cs_taps(cs, p, idx, wt);
	vec3 r = v3(0, 0, 0);
	for (int k = 0; k < 4; k++)
		r = v3_add(r, v3_scale(v3_load(fld3 + 3 * idx[k]), wt[k]));
	return v3_tangent(r, p);
}

/* Area of spherical triangle with unit-vector vertices. */
static double sph_tri_area(vec3 a, vec3 b, vec3 c)
{
	double num = fabs(v3_dot(a, v3_cross(b, c)));
	double den = 1.0 + v3_dot(a, b) + v3_dot(b, c) + v3_dot(c, a);
	return 2.0 * atan2(num, den);
}

/* ---- Spherical Voronoi cell construction ----
 * The bisector of two points on the sphere is a great circle, which is a
 * straight line in the gnomonic projection centered on the cell. So each
 * Voronoi cell is computed exactly by clipping a polygon by half-planes in
 * that projection. */

#define MAXPV 64
typedef struct { double a, b; int lab; } PVert; /* lab = label of edge starting here */

static int clip_poly(PVert *in, int n, PVert *out, double A, double B, double C, int lab)
{
	int m = 0;
	for (int k = 0; k < n; k++) {
		PVert *v0 = &in[k], *v1 = &in[(k + 1) % n];
		double f0 = A * v0->a + B * v0->b + C;
		double f1 = A * v1->a + B * v1->b + C;
		int in0 = f0 >= 0, in1 = f1 >= 0;
		if (in0) {
			if (m < MAXPV) out[m++] = *v0;
			if (!in1) {
				double t = f0 / (f0 - f1);
				PVert I = { v0->a + t * (v1->a - v0->a), v0->b + t * (v1->b - v0->b), lab };
				if (m < MAXPV) out[m++] = I;
			}
		} else if (in1) {
			double t = f0 / (f0 - f1);
			PVert I = { v0->a + t * (v1->a - v0->a), v0->b + t * (v1->b - v0->b), v0->lab };
			if (m < MAXPV) out[m++] = I;
		}
	}
	return m;
}

typedef struct { int nb; double len; } Link;

/* Compute Voronoi links (neighbor, edge length) and area for cell c. */
static int voronoi_cell(const CubeSphere *cs, int c, Link *links, double *area, int *open)
{
	int N = cs->N;
	int f = c / (N * N), rem = c % (N * N);
	int j = rem / N, i = rem % N;
	vec3 p = cs_pos(cs, c);
	vec3 e1 = v3_normalize(v3_tangent(cs_face_right[f], p));
	vec3 e2 = v3_cross(p, e1);

	int cand[24], nc = 0;
	for (int dj = -2; dj <= 2; dj++)
		for (int di = -2; di <= 2; di++) {
			if (!di && !dj)
				continue;
			int q = tap_index(cs, f, i + di, j + dj);
			if (q == c)
				continue;
			int dup = 0;
			for (int k = 0; k < nc; k++)
				if (cand[k] == q)
					dup = 1;
			if (!dup)
				cand[nc++] = q;
		}

	PVert bufA[MAXPV], bufB[MAXPV];
	double S = 4 * cs->h;
	PVert *poly = bufA, *tmp = bufB;
	int n = 4;
	poly[0] = (PVert) { -S, -S, -1 };
	poly[1] = (PVert) { S, -S, -1 };
	poly[2] = (PVert) { S, S, -1 };
	poly[3] = (PVert) { -S, S, -1 };
	for (int k = 0; k < nc; k++) {
		vec3 q = cs_pos(cs, cand[k]);
		vec3 m = v3_sub(p, q);
		n = clip_poly(poly, n, tmp, v3_dot(e1, m), v3_dot(e2, m), v3_dot(p, m), cand[k]);
		PVert *t = poly; poly = tmp; tmp = t;
	}

	/* convert to sphere, accumulate links and area */
	int nl = 0;
	double ar = 0;
	*open = 0;
	for (int k = 0; k < n; k++) {
		PVert *v0 = &poly[k], *v1 = &poly[(k + 1) % n];
		vec3 V0 = v3_normalize(v3_add(p, v3_add(v3_scale(e1, v0->a), v3_scale(e2, v0->b))));
		vec3 V1 = v3_normalize(v3_add(p, v3_add(v3_scale(e1, v1->a), v3_scale(e2, v1->b))));
		ar += sph_tri_area(p, V0, V1);
		if (v0->lab < 0) {
			*open = 1;
			continue;
		}
		double len = v3_angle(V0, V1);
		int found = 0;
		for (int l = 0; l < nl; l++)
			if (links[l].nb == v0->lab) {
				links[l].len += len;
				found = 1;
			}
		if (!found && nl < 24) {
			links[nl].nb = v0->lab;
			links[nl].len = len;
			nl++;
		}
	}
	*area = ar;
	return nl;
}

static int link_cmp(const void *a, const void *b)
{
	double la = ((const Link *) a)->len, lb = ((const Link *) b)->len;
	return la < lb ? 1 : (la > lb ? -1 : 0);
}

int cs_init(CubeSphere *cs, int N)
{
	memset(cs, 0, sizeof(*cs));
	cs->N = N;
	cs->ncells = 6 * N * N;
	cs->h = (M_PI / 2) / N;
	int n = cs->ncells;
	cs->pos = malloc(sizeof(float) * 3 * n);
	cs->area = malloc(sizeof(float) * n);
	cs->nnb = calloc(n, 1);
	cs->nbr = malloc(sizeof(int) * CS_MAXNB * n);
	cs->w = malloc(sizeof(float) * CS_MAXNB * n);
	cs->gc = malloc(sizeof(float) * 3 * CS_MAXNB * n);
	float *elen = malloc(sizeof(float) * CS_MAXNB * n);
	cs->elen = malloc(sizeof(float) * CS_MAXNB * n);
	if (!cs->pos || !cs->area || !cs->nnb || !cs->nbr || !cs->w || !cs->gc || !elen || !cs->elen) {
		fprintf(stderr, "cs_init: out of memory\n");
		return -1;
	}
	double h = cs->h;

	#pragma omp parallel for schedule(static)
	for (int c = 0; c < n; c++) {
		int f = c / (N * N), rem = c % (N * N);
		v3_store(cs->pos + 3 * c, cell_dir(cs, f, rem % N, rem / N));
	}

	/* Voronoi cells; drop negligible edges (numerical slivers at 4-way junctions) */
	double total = 0;
	int open = 0;
	double minlen = 1e-4 * h;
	#pragma omp parallel for schedule(static) reduction(+:total,open)
	for (int c = 0; c < n; c++) {
		Link links[24];
		double ar;
		int op;
		int nl = voronoi_cell(cs, c, links, &ar, &op);
		open += op;
		total += ar;
		cs->area[c] = (float) ar;
		qsort(links, nl, sizeof(Link), link_cmp);
		int m = 0;
		for (int l = 0; l < nl && m < CS_MAXNB; l++) {
			if (links[l].len < minlen)
				break;
			cs->nbr[CS_MAXNB * c + m] = links[l].nb;
			elen[CS_MAXNB * c + m] = (float) links[l].len;
			m++;
		}
		cs->nnb[c] = (unsigned char) m;
	}
	cs->total_area = total;
	cs->open_cells = open;

	/* Symmetrize: keep only reciprocal links; average edge lengths. */
	float *len2 = malloc(sizeof(float) * CS_MAXNB * n);
	int asym = 0;
	#pragma omp parallel for schedule(static) reduction(+:asym)
	for (int c = 0; c < n; c++) {
		for (int k = 0; k < cs->nnb[c]; k++) {
			int jc = cs->nbr[CS_MAXNB * c + k];
			int found = -1;
			for (int kk = 0; kk < cs->nnb[jc]; kk++)
				if (cs->nbr[CS_MAXNB * jc + kk] == c)
					found = kk;
			double l = elen[CS_MAXNB * c + k];
			if (found < 0) {
				if (l > 1e-2 * h)
					asym++;
				len2[CS_MAXNB * c + k] = -1.0f; /* drop */
			} else {
				len2[CS_MAXNB * c + k] = (float) (0.5 * (l + elen[CS_MAXNB * jc + found]));
			}
		}
	}
	cs->asym_count = asym;

	/* Final links: Laplacian weights and least-squares gradient coefficients. */
	#pragma omp parallel for schedule(static)
	for (int c = 0; c < n; c++) {
		vec3 p = cs_pos(cs, c);
		int m = 0;
		int nb[CS_MAXNB];
		double ln[CS_MAXNB];
		for (int k = 0; k < cs->nnb[c]; k++) {
			if (len2[CS_MAXNB * c + k] < 0)
				continue;
			nb[m] = cs->nbr[CS_MAXNB * c + k];
			ln[m] = len2[CS_MAXNB * c + k];
			m++;
		}
		cs->nnb[c] = (unsigned char) m;
		int f = c / (N * N);
		vec3 e1 = v3_normalize(v3_tangent(cs_face_right[f], p));
		vec3 e2 = v3_cross(p, e1);
		double M00 = 0, M01 = 0, M11 = 0;
		double tx[CS_MAXNB], ty[CS_MAXNB], wk[CS_MAXNB];
		for (int k = 0; k < m; k++) {
			vec3 q = cs_pos(cs, nb[k]);
			double d = v3_angle(p, q);
			cs->nbr[CS_MAXNB * c + k] = nb[k];
			cs->w[CS_MAXNB * c + k] = (float) (ln[k] / d);
			cs->elen[CS_MAXNB * c + k] = (float) ln[k];
			vec3 t = v3_tangent(v3_sub(q, p), p);
			tx[k] = v3_dot(t, e1);
			ty[k] = v3_dot(t, e2);
			wk[k] = ln[k] / (d * d * d);
			M00 += wk[k] * tx[k] * tx[k];
			M01 += wk[k] * tx[k] * ty[k];
			M11 += wk[k] * ty[k] * ty[k];
		}
		double det = M00 * M11 - M01 * M01;
		double i00 = M11 / det, i01 = -M01 / det, i11 = M00 / det;
		for (int k = 0; k < m; k++) {
			double c1 = wk[k] * (i00 * tx[k] + i01 * ty[k]);
			double c2 = wk[k] * (i01 * tx[k] + i11 * ty[k]);
			v3_store(cs->gc + 3 * (CS_MAXNB * c + k), v3_add(v3_scale(e1, c1), v3_scale(e2, c2)));
		}
	}
	free(len2);
	free(elen);
	if (asym)
		fprintf(stderr, "cs_init: warning: %d non-reciprocal Voronoi links\n", asym);
	if (open)
		fprintf(stderr, "cs_init: warning: %d open Voronoi cells\n", open);
	return 0;
}

void cs_free(CubeSphere *cs)
{
	free(cs->pos);
	free(cs->area);
	free(cs->nnb);
	free(cs->nbr);
	free(cs->w);
	free(cs->gc);
	free(cs->elen);
	memset(cs, 0, sizeof(*cs));
}
