/* 3D simplex noise with analytic gradients, and multi-octave curl noise. */
#ifndef NOISE_H
#define NOISE_H

#include "vec3.h"

typedef struct {
	unsigned char perm[512];
} Noise;

typedef struct {
	double scale;       /* base frequency on the unit sphere */
	double lacunarity;  /* frequency multiplier per octave */
	double gain;        /* amplitude multiplier per octave */
	int octaves;
} CurlParams;

void noise_init(Noise *nz, unsigned int seed);
/* Returns noise value; writes the analytic gradient to g (may be NULL). */
double simplex3(const Noise *nz, vec3 p, vec3 *g);
/* Divergence-free 3D curl noise (not projected onto the sphere). */
vec3 curl_noise(const Noise *nz, vec3 p, const CurlParams *cp);

#endif
