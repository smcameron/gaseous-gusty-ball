#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "particles.h"

int particles_init(Particles *P, int n)
{
	P->n = n;
	P->pos = malloc(sizeof(float) * 3 * n);
	P->prev = malloc(sizeof(float) * 3 * n);
	P->col = malloc(sizeof(float) * 4 * n);
	if (!P->pos || !P->prev || !P->col) {
		fprintf(stderr, "particles: out of memory\n");
		return -1;
	}
	const double golden = M_PI * (3.0 - sqrt(5.0));
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		double y = 1.0 - 2.0 * (i + 0.5) / n;
		double r = sqrt(1.0 - y * y);
		double phi = golden * i;
		vec3 p = v3(r * cos(phi), y, r * sin(phi));
		v3_store(P->pos + 3 * i, p);
		v3_store(P->prev + 3 * i, p);
	}
	for (int i = 0; i < 4 * n; i++)
		P->col[i] = 1.0f;
	return 0;
}

void particles_free(Particles *P)
{
	free(P->pos);
	free(P->prev);
	free(P->col);
	memset(P, 0, sizeof(*P));
}

void particles_color_from_image(Particles *P, const float *img, int w, int h)
{
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < P->n; i++) {
		double lon, lat;
		dir_to_lonlat(v3_load(P->pos + 3 * i), &lon, &lat);
		double x = (lon + M_PI) / (2 * M_PI) * w - 0.5;
		double y = (M_PI / 2 - lat) / M_PI * h - 0.5;
		int x0 = (int) floor(x), y0 = (int) floor(y);
		double fx = x - x0, fy = y - y0;
		int xs[2] = { ((x0 % w) + w) % w, (((x0 + 1) % w) + w) % w };
		int ys[2] = { y0 < 0 ? 0 : (y0 >= h ? h - 1 : y0), y0 + 1 < 0 ? 0 : (y0 + 1 >= h ? h - 1 : y0 + 1) };
		double wts[4] = { (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy };
		for (int c = 0; c < 4; c++) {
			double v = 0;
			for (int k = 0; k < 4; k++)
				v += wts[k] * img[((size_t) ys[k >> 1] * w + xs[k & 1]) * 4 + c];
			P->col[4 * i + c] = (float) v;
		}
	}
}

void particles_advect(Particles *P, const CubeSphere *cs, const float *vel, double dt)
{
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < P->n; i++) {
		vec3 p = v3_load(P->pos + 3 * i);
		v3_store(P->prev + 3 * i, p);
		vec3 u1 = cs_sample_vec(cs, vel, p);
		vec3 pm = sphere_move(p, u1, 0.5 * dt);
		vec3 u2 = v3_tangent(cs_sample_vec(cs, vel, pm), p);
		v3_store(P->pos + 3 * i, sphere_move(p, u2, dt));
	}
}
