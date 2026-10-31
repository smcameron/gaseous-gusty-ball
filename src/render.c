#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "render.h"

#define CHUNK 262144

int render_init(Render *r, int R, float radius, float alpha)
{
	memset(r, 0, sizeof(*r));
	r->R = R;
	r->radius = radius < 0.75f ? 0.75f : radius;
	r->alpha = alpha;
	r->g = 1.0;
	r->buf = calloc((size_t) 6 * R * R * 4, sizeof(float));
	if (!r->buf) {
		fprintf(stderr, "render: out of memory\n");
		return -1;
	}
	int sh = 32;
	if (sh < (int) ceilf(r->radius) + 1)
		sh = (int) ceilf(r->radius) + 1;
	if (sh > R)
		sh = R;
	r->strip_h = sh;
	r->nstrips = (R + sh - 1) / sh;
	r->nbins = 6 * r->nstrips;
	r->maxthreads = omp_get_max_threads();
	r->tr = calloc(r->maxthreads, sizeof(ThreadRecs));
	for (int t = 0; t < r->maxthreads; t++) {
		r->tr[t].hist = calloc(r->nbins, sizeof(size_t));
		r->tr[t].cursor = calloc(r->nbins, sizeof(size_t));
	}
	r->bin_start = calloc(r->nbins + 1, sizeof(size_t));
	return 0;
}

void render_free(Render *r)
{
	for (int t = 0; t < r->maxthreads; t++) {
		free(r->tr[t].v);
		free(r->tr[t].hist);
		free(r->tr[t].cursor);
	}
	free(r->tr);
	free(r->sorted);
	free(r->bin_start);
	free(r->buf);
	memset(r, 0, sizeof(*r));
}

void render_fade(Render *r, double fade)
{
	r->g *= (1.0 - fade);
	if (r->g < 1e-4) {
		size_t n = (size_t) 6 * r->R * r->R * 4;
		float g = (float) r->g;
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < n; i++)
			r->buf[i] *= g;
		r->g = 1.0;
	}
}

static inline void push_rec(ThreadRecs *t, float px, float py, uint32_t pidx, uint32_t bin)
{
	if (t->n == t->cap) {
		t->cap = t->cap ? t->cap * 2 : 65536;
		t->v = realloc(t->v, t->cap * sizeof(SplatRec));
		if (!t->v) {
			fprintf(stderr, "render: out of memory\n");
			exit(1);
		}
	}
	SplatRec *s = &t->v[t->n++];
	s->px = px;
	s->py = py;
	s->pidx = pidx;
	s->bin = bin;
	t->hist[bin]++;
}

/* Emit splat records for one direction on every face it touches. */
static void emit_point(const Render *r, ThreadRecs *t, vec3 d, uint32_t pidx)
{
	int R = r->R;
	float rad = r->radius;
	for (int f = 0; f < 6; f++) {
		double dc = v3_dot(d, cs_face_center[f]);
		if (dc < 0.5)
			continue;
		double u = v3_dot(d, cs_face_right[f]) / dc;
		double v = v3_dot(d, cs_face_up[f]) / dc;
		float px = (float) ((u + 1.0) * 0.5 * R - 0.5);
		float py = (float) ((1.0 - v) * 0.5 * R - 0.5);
		int xlo = (int) ceilf(px - rad), xhi = (int) floorf(px + rad);
		int ylo = (int) ceilf(py - rad), yhi = (int) floorf(py + rad);
		if (xlo < 0) xlo = 0;
		if (xhi > R - 1) xhi = R - 1;
		if (ylo < 0) ylo = 0;
		if (yhi > R - 1) yhi = R - 1;
		if (xlo > xhi || ylo > yhi)
			continue;
		int s0 = ylo / r->strip_h, s1 = yhi / r->strip_h;
		for (int s = s0; s <= s1; s++)
			push_rec(t, px, py, pidx, (uint32_t) (f * r->nstrips + s));
	}
}

static void draw_bin(Render *r, const Particles *P, int b)
{
	int R = r->R;
	int f = b / r->nstrips, s = b % r->nstrips;
	int y0s = s * r->strip_h;
	int y1s = y0s + r->strip_h - 1;
	if (y1s > R - 1)
		y1s = R - 1;
	float rad = r->radius, invrad = 1.0f / rad;
	float invg = (float) (1.0 / r->g);
	float *face = r->buf + (size_t) f * R * R * 4;
	for (size_t k = r->bin_start[b]; k < r->bin_start[b + 1]; k++) {
		const SplatRec *sp = &r->sorted[k];
		const float *col = P->col + 4 * sp->pidx;
		float w0 = r->alpha * col[3];
		float px = sp->px, py = sp->py;
		int ylo = (int) ceilf(py - rad), yhi = (int) floorf(py + rad);
		int xlo = (int) ceilf(px - rad), xhi = (int) floorf(px + rad);
		if (ylo < y0s) ylo = y0s;
		if (yhi > y1s) yhi = y1s;
		if (xlo < 0) xlo = 0;
		if (xhi > R - 1) xhi = R - 1;
		for (int y = ylo; y <= yhi; y++) {
			float wy = 1.0f - fabsf(y - py) * invrad;
			if (wy <= 0)
				continue;
			float *row = face + (size_t) y * R * 4;
			for (int x = xlo; x <= xhi; x++) {
				float wx = 1.0f - fabsf(x - px) * invrad;
				if (wx <= 0)
					continue;
				float w = w0 * wx * wy;
				float keep = 1.0f - w, add = w * invg;
				float *d = row + 4 * x;
				d[0] = d[0] * keep + add * col[0];
				d[1] = d[1] * keep + add * col[1];
				d[2] = d[2] * keep + add * col[2];
				d[3] = d[3] * keep + add;
			}
		}
	}
}

void render_deposit(Render *r, const Particles *P)
{
	/* angular sample spacing along each trail segment (~0.35..0.7 texel) */
	double spacing = 0.7 / r->R;
	int nbins = r->nbins;
	for (int start = 0; start < P->n; start += CHUNK) {
		int end = start + CHUNK < P->n ? start + CHUNK : P->n;
		int nthreads = 1;
		for (int t = 0; t < r->maxthreads; t++) {
			r->tr[t].n = 0;
			memset(r->tr[t].hist, 0, sizeof(size_t) * nbins);
		}
		#pragma omp parallel
		{
			int t = omp_get_thread_num(), nt = omp_get_num_threads();
			if (t == 0)
				nthreads = nt;
			ThreadRecs *me = &r->tr[t];
			int lo = start + (int) ((long long) (end - start) * t / nt);
			int hi = start + (int) ((long long) (end - start) * (t + 1) / nt);
			for (int i = lo; i < hi; i++) {
				vec3 a = v3_load(P->prev + 3 * i), b = v3_load(P->pos + 3 * i);
				double ang = v3_angle(a, b);
				int ns = (int) ceil(ang / spacing);
				if (ns < 1) ns = 1;
				if (ns > 4096) ns = 4096;
				for (int k = 1; k <= ns; k++) {
					double tt = (double) k / ns;
					vec3 d = v3_normalize(v3_add(v3_scale(a, 1 - tt), v3_scale(b, tt)));
					emit_point(r, me, d, (uint32_t) i);
				}
			}
		}
		/* counting sort offsets: bin-major, thread-minor (deterministic order) */
		size_t acc = 0;
		for (int b = 0; b < nbins; b++) {
			r->bin_start[b] = acc;
			for (int t = 0; t < nthreads; t++) {
				r->tr[t].cursor[b] = acc;
				acc += r->tr[t].hist[b];
			}
		}
		r->bin_start[nbins] = acc;
		if (acc > r->sorted_cap) {
			free(r->sorted);
			r->sorted_cap = acc + acc / 4;
			r->sorted = malloc(r->sorted_cap * sizeof(SplatRec));
			if (!r->sorted) {
				fprintf(stderr, "render: out of memory\n");
				exit(1);
			}
		}
		#pragma omp parallel for schedule(static)
		for (int t = 0; t < nthreads; t++) {
			ThreadRecs *me = &r->tr[t];
			for (size_t k = 0; k < me->n; k++)
				r->sorted[me->cursor[me->v[k].bin]++] = me->v[k];
		}
		#pragma omp parallel for schedule(dynamic, 1)
		for (int b = 0; b < nbins; b++)
			draw_bin(r, P, b);
	}
}

void render_texel(const Render *r, int face, int x, int y, float out[4])
{
	const float *d = r->buf + (((size_t) face * r->R + y) * r->R + x) * 4;
	float g = (float) r->g;
	for (int c = 0; c < 4; c++)
		out[c] = d[c] * g;
}

void render_sample_dir(const Render *r, double dx, double dy, double dz, float out[4])
{
	vec3 d = v3(dx, dy, dz);
	int f = cs_face_of(d);
	double dc = v3_dot(d, cs_face_center[f]);
	double u = v3_dot(d, cs_face_right[f]) / dc;
	double v = v3_dot(d, cs_face_up[f]) / dc;
	int R = r->R;
	double px = (u + 1.0) * 0.5 * R - 0.5, py = (1.0 - v) * 0.5 * R - 0.5;
	if (px < 0) px = 0;
	if (py < 0) py = 0;
	if (px > R - 1) px = R - 1;
	if (py > R - 1) py = R - 1;
	int x0 = (int) px, y0 = (int) py;
	int x1 = x0 + 1 < R ? x0 + 1 : x0, y1 = y0 + 1 < R ? y0 + 1 : y0;
	float fx = (float) (px - x0), fy = (float) (py - y0);
	float a[4], b[4], c[4], e[4];
	render_texel(r, f, x0, y0, a);
	render_texel(r, f, x1, y0, b);
	render_texel(r, f, x0, y1, c);
	render_texel(r, f, x1, y1, e);
	for (int k = 0; k < 4; k++)
		out[k] = (1 - fy) * ((1 - fx) * a[k] + fx * b[k]) + fy * ((1 - fx) * c[k] + fx * e[k]);
}

static inline unsigned char to8(float v)
{
	if (v <= 0) return 0;
	if (v >= 1) return 255;
	return (unsigned char) (v * 255.0f + 0.5f);
}

void render_pixel8(const float c[4], const float *bg, unsigned char *out)
{
	float a = c[3];
	if (a > 1) a = 1;
	if (a < 0) a = 0;
	if (!bg) {
		if (a > 1e-6f) {
			out[0] = to8(c[0] / a);
			out[1] = to8(c[1] / a);
			out[2] = to8(c[2] / a);
		} else {
			out[0] = out[1] = out[2] = 0;
		}
		out[3] = to8(a);
	} else {
		out[0] = to8(c[0] + bg[0] * (1 - a));
		out[1] = to8(c[1] + bg[1] * (1 - a));
		out[2] = to8(c[2] + bg[2] * (1 - a));
	}
}
