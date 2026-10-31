/* Fading, alpha-blended particle-trail renderer into 6 cubemap faces.
 *
 * Faces use the gnomonic (standard cubemap) projection with the face basis
 * from cubesphere.h. Pixel (col, row): row 0 is the top of the image.
 *
 * Buffers hold premultiplied RGBA, stored divided by a global decay factor g
 * so that fading costs O(1) per step (with occasional renormalization).
 */
#ifndef RENDER_H
#define RENDER_H

#include <stddef.h>
#include <stdint.h>

#include "particles.h"

typedef struct {
	float px, py;
	uint32_t pidx;
	uint32_t bin;
} SplatRec;

typedef struct {
	SplatRec *v;
	size_t n, cap;
	size_t *hist;   /* [nbins] */
	size_t *cursor; /* [nbins] */
} ThreadRecs;

typedef struct {
	int R;
	float *buf;     /* [6*R*R*4] scaled premultiplied RGBA */
	double g;       /* global decay factor */
	float radius;   /* splat radius in texels */
	float alpha;    /* per-splat opacity */
	int strip_h, nstrips, nbins;
	int maxthreads;
	ThreadRecs *tr;
	SplatRec *sorted;
	size_t sorted_cap;
	size_t *bin_start; /* [nbins+1] */
} Render;

int render_init(Render *r, int R, float radius, float alpha);
void render_free(Render *r);
void render_fade(Render *r, double fade);
void render_deposit(Render *r, const Particles *P);
/* Actual premultiplied RGBA of a texel. */
void render_texel(const Render *r, int face, int x, int y, float out[4]);
/* Bilinear sample in direction d (premultiplied RGBA). */
void render_sample_dir(const Render *r, double dx, double dy, double dz, float out[4]);
/* Convert premultiplied RGBA to 8-bit: bg == NULL -> RGBA (un-premultiplied),
 * else RGB composited over bg. */
void render_pixel8(const float c[4], const float *bg, unsigned char *out);

#endif
