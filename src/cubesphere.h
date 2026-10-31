/* Equiangular cubed-sphere grid: geometry, neighbor tables and sampling.
 *
 * The face basis (center / right / up) is the same as the output "cross"
 * cubemap layout:
 *
 *        +---+
 *        | 4 |
 *        +---+---+---+---+
 *        | 0 | 1 | 2 | 3 |
 *        +---+---+---+---+
 *        | 5 |
 *        +---+
 *
 * Face 0 = +Z, 1 = +X, 2 = -Z, 3 = -X, 4 = +Y (north), 5 = -Y (south).
 * Within a face, cell (i, j) has i increasing to the "right" and j
 * increasing "up".
 *
 * Differential operators use the spherical Voronoi diagram of the cell
 * centers (computed exactly at init): the Laplacian is the symmetric
 * Voronoi finite-volume Laplacian, and gradients use least-squares
 * coefficients over the Voronoi neighbors. Both are consistent on the
 * skewed equiangular grid and across the cube edges.
 */
#ifndef CUBESPHERE_H
#define CUBESPHERE_H

#include "vec3.h"

extern const vec3 cs_face_center[6];
extern const vec3 cs_face_right[6];
extern const vec3 cs_face_up[6];

#define CS_MAXNB 8

typedef struct {
	int N;          /* cells per face edge */
	int ncells;     /* 6*N*N */
	double h;       /* angular cell width (pi/2)/N */
	float *pos;     /* [ncells*3] unit cell-center positions */
	float *area;    /* [ncells] Voronoi cell area */
	unsigned char *nnb; /* [ncells] number of Voronoi neighbors */
	int *nbr;       /* [ncells*CS_MAXNB] neighbor indices */
	float *w;       /* [ncells*CS_MAXNB] symmetric Laplacian weights edge_len / center_dist */
	float *elen;    /* [ncells*CS_MAXNB] Voronoi edge length (radians) */
	float *gc;      /* [ncells*CS_MAXNB*3] least-squares gradient coefficients:
	                 * grad f(i) ~= sum_k gc_k * (f(nbr_k) - f(i)) */
	double total_area;
	int asym_count; /* number of non-reciprocal Voronoi links (should be 0) */
	int open_cells; /* Voronoi cells not closed by candidates (should be 0) */
} CubeSphere;

int cs_init(CubeSphere *cs, int N);
void cs_free(CubeSphere *cs);

int cs_face_of(vec3 d);
/* Direction for equiangular face angles a, b in [-pi/4, pi/4] (may extend beyond). */
vec3 cs_face_ab_to_dir(int f, double a, double b);
/* Continuous cell coordinates; cell centers lie at integer x, y. */
void cs_locate(const CubeSphere *cs, vec3 d, int *f, double *x, double *y);
int cs_cell_of(const CubeSphere *cs, vec3 d);
/* Bilinear taps (4 indices, 4 weights) for sampling at direction p. */
void cs_taps(const CubeSphere *cs, vec3 p, int idx[4], double wt[4]);
double cs_sample_scalar(const CubeSphere *cs, const float *fld, vec3 p);
/* Bilinear sample of a tangent vector field; result is projected onto the
 * tangent plane at p. */
vec3 cs_sample_vec(const CubeSphere *cs, const float *fld3, vec3 p);

static inline int cs_index(const CubeSphere *cs, int f, int i, int j)
{
	return (f * cs->N + j) * cs->N + i;
}

static inline vec3 cs_pos(const CubeSphere *cs, int c) { return v3_load(cs->pos + 3 * c); }

#endif
