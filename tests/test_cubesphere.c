#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cubesphere.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static double frand(unsigned *s)
{
	*s = *s * 1664525u + 1013904223u;
	return (*s >> 8) / 16777216.0;
}

/* gnomonic projection onto face f as used by the output layout */
static void project(int f, vec3 d, double *u, double *v)
{
	double dc = v3_dot(d, cs_face_center[f]);
	*u = v3_dot(d, cs_face_right[f]) / dc;
	*v = v3_dot(d, cs_face_up[f]) / dc;
}

int main(void)
{
	CubeSphere cs;
	int N = 64;
	cs_init(&cs, N);

	/* round trip */
	unsigned s = 12345;
	double maxerr = 0;
	for (int k = 0; k < 100000; k++) {
		vec3 d = v3_normalize(v3(frand(&s) * 2 - 1, frand(&s) * 2 - 1, frand(&s) * 2 - 1));
		int f;
		double x, y;
		cs_locate(&cs, d, &f, &x, &y);
		double a = -M_PI / 4 + (x + 0.5) * cs.h, b = -M_PI / 4 + (y + 0.5) * cs.h;
		vec3 d2 = cs_face_ab_to_dir(f, a, b);
		double e = v3_len(v3_sub(d, d2));
		if (e > maxerr) maxerr = e;
	}
	printf("round trip max error: %g\n", maxerr);
	CHECK(maxerr < 1e-9, "round trip error %g", maxerr);

	/* cell centers locate back to themselves */
	int bad = 0;
	for (int c = 0; c < cs.ncells; c++)
		if (cs_cell_of(&cs, cs_pos(&cs, c)) != c)
			bad++;
	CHECK(bad == 0, "%d cell centers don't map back to their cell", bad);

	/* neighbor symmetry */
	printf("non-reciprocal neighbor links: %d\n", cs.asym_count);
	CHECK(cs.asym_count == 0, "asymmetric neighbor links: %d", cs.asym_count);
	int selfn = 0;
	for (int c = 0; c < cs.ncells; c++)
		for (int k = 0; k < cs.nnb[c]; k++)
			if (cs.nbr[CS_MAXNB * c + k] == c)
				selfn++;
	CHECK(selfn == 0, "%d cells are their own neighbor", selfn);
	CHECK(cs.open_cells == 0, "%d open Voronoi cells", cs.open_cells);
	int nbhist[CS_MAXNB + 1] = {0};
	for (int c = 0; c < cs.ncells; c++)
		nbhist[cs.nnb[c]]++;
	printf("Voronoi neighbor count histogram:");
	for (int k = 0; k <= CS_MAXNB; k++)
		if (nbhist[k]) printf(" %d:%d", k, nbhist[k]);
	printf("\n");
	CHECK(nbhist[0] + nbhist[1] + nbhist[2] == 0, "cells with < 3 neighbors");

	/* area */
	double relerr = fabs(cs.total_area - 4 * M_PI) / (4 * M_PI);
	printf("total area = %.8f (4pi = %.8f), rel err %g\n", cs.total_area, 4 * M_PI, relerr);
	CHECK(relerr < 1e-4, "area rel err %g", relerr);

	/* Output-layout adjacency: an edge point of face f in direction (L/R/top/bottom)
	 * must lie on the specified edge of the expected neighbor face. */
	struct { int f; int edge; int nf; int nedge; } adj[] = {
		/* edge: 0=left(u=-1) 1=right(u=+1) 2=top(v=+1) 3=bottom(v=-1) */
		{0, 1, 1, 0}, {1, 1, 2, 0}, {2, 1, 3, 0}, {3, 1, 0, 0},
		{0, 2, 4, 3}, {0, 3, 5, 2},
		{1, 2, 4, -1}, {2, 2, 4, -1}, {3, 2, 4, -1},
		{1, 3, 5, -1}, {2, 3, 5, -1}, {3, 3, 5, -1},
	};
	for (size_t k = 0; k < sizeof(adj) / sizeof(adj[0]); k++) {
		int f = adj[k].f, nf = adj[k].nf;
		for (int t = 1; t < 10; t++) {
			double par = -1 + 2.0 * t / 10;
			double u, v;
			switch (adj[k].edge) {
			case 0: u = -1; v = par; break;
			case 1: u = 1; v = par; break;
			case 2: u = par; v = 1; break;
			default: u = par; v = -1; break;
			}
			vec3 d = v3_normalize(v3_add(cs_face_center[f],
				v3_add(v3_scale(cs_face_right[f], u), v3_scale(cs_face_up[f], v))));
			double nu, nv;
			project(nf, d, &nu, &nv);
			int ok;
			switch (adj[k].nedge) {
			case 0: ok = fabs(nu + 1) < 1e-9 && fabs(nv - v) < 1e-9; break; /* same row */
			case 2: ok = fabs(nv - 1) < 1e-9 && fabs(nu - u) < 1e-9; break; /* same column */
			case 3: ok = fabs(nv + 1) < 1e-9 && fabs(nu - u) < 1e-9; break;
			default: ok = fabs(fabs(nu) - 1) < 1e-9 || fabs(fabs(nv) - 1) < 1e-9; break;
			}
			CHECK(ok, "layout adjacency face %d edge %d -> face %d: (%g,%g)", f, adj[k].edge, nf, nu, nv);
		}
	}

	cs_free(&cs);
	if (failures) {
		printf("test_cubesphere: %d failures\n", failures);
		return 1;
	}
	printf("test_cubesphere: OK\n");
	return 0;
}
