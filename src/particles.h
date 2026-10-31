/* Passive colored tracer particles on the unit sphere. */
#ifndef PARTICLES_H
#define PARTICLES_H

#include "cubesphere.h"

typedef struct {
	int n;
	float *pos;  /* [n*3] current positions */
	float *prev; /* [n*3] positions before the last advection step */
	float *col;  /* [n*4] RGBA colors (fixed) */
} Particles;

/* Evenly distributed (Fibonacci lattice) initial positions. */
int particles_init(Particles *P, int n);
void particles_free(Particles *P);
/* Equirectangular sampling: image x -> longitude (wraps), image y -> latitude
 * (row 0 = north pole). */
void particles_color_from_image(Particles *P, const float *img, int w, int h);
/* RK2 (midpoint) advection along great circles; saves prev positions. */
void particles_advect(Particles *P, const CubeSphere *cs, const float *vel, double dt);

#endif
