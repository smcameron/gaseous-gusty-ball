#include <math.h>
#include <stdint.h>

#include "noise.h"

static const double grad3[12][3] = {
	{1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
	{1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
	{0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1},
};

void noise_init(Noise *nz, unsigned int seed)
{
	unsigned char p[256];
	uint64_t s = 0x9E3779B97F4A7C15ULL ^ ((uint64_t) seed * 0xD1B54A32D192ED03ULL);
	for (int i = 0; i < 256; i++)
		p[i] = (unsigned char) i;
	for (int i = 255; i > 0; i--) {
		/* xorshift64* */
		s ^= s >> 12;
		s ^= s << 25;
		s ^= s >> 27;
		uint64_t r = s * 2685821657736338717ULL;
		int j = (int) ((r >> 33) % (uint64_t) (i + 1));
		unsigned char t = p[i];
		p[i] = p[j];
		p[j] = t;
	}
	for (int i = 0; i < 512; i++)
		nz->perm[i] = p[i & 255];
}

double simplex3(const Noise *nz, vec3 pt, vec3 *g)
{
	const double F3 = 1.0 / 3.0, G3 = 1.0 / 6.0;
	const unsigned char *perm = nz->perm;
	double x = pt.x, y = pt.y, z = pt.z;
	double s = (x + y + z) * F3;
	int i = (int) floor(x + s), j = (int) floor(y + s), k = (int) floor(z + s);
	double t = (i + j + k) * G3;
	double x0 = x - (i - t), y0 = y - (j - t), z0 = z - (k - t);
	int i1, j1, k1, i2, j2, k2;
	if (x0 >= y0) {
		if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
		else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
		else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
	} else {
		if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
		else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
		else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
	}
	double xs[4] = { x0, x0 - i1 + G3, x0 - i2 + 2 * G3, x0 - 1 + 3 * G3 };
	double ys[4] = { y0, y0 - j1 + G3, y0 - j2 + 2 * G3, y0 - 1 + 3 * G3 };
	double zs[4] = { z0, z0 - k1 + G3, z0 - k2 + 2 * G3, z0 - 1 + 3 * G3 };
	int ii = i & 255, jj = j & 255, kk = k & 255;
	int gi[4];
	gi[0] = perm[ii + perm[jj + perm[kk]]] % 12;
	gi[1] = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
	gi[2] = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
	gi[3] = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12;

	double n = 0, dx = 0, dy = 0, dz = 0;
	for (int c = 0; c < 4; c++) {
		double tt = 0.5 - xs[c] * xs[c] - ys[c] * ys[c] - zs[c] * zs[c];
		if (tt <= 0)
			continue;
		const double *gr = grad3[gi[c]];
		double gd = gr[0] * xs[c] + gr[1] * ys[c] + gr[2] * zs[c];
		double t2 = tt * tt, t4 = t2 * t2, t3 = t2 * tt;
		n += t4 * gd;
		dx += t4 * gr[0] - 8.0 * t3 * gd * xs[c];
		dy += t4 * gr[1] - 8.0 * t3 * gd * ys[c];
		dz += t4 * gr[2] - 8.0 * t3 * gd * zs[c];
	}
	const double scale = 72.0;
	if (g)
		*g = v3(dx * scale, dy * scale, dz * scale);
	return n * scale;
}

vec3 curl_noise(const Noise *nz, vec3 p, const CurlParams *cp)
{
	static const vec3 off[3] = {
		{ 0.0, 0.0, 0.0 }, { 31.416, -47.853, 12.793 }, { -93.11, 63.2, 109.7 },
	};
	vec3 c = v3(0, 0, 0);
	double f = cp->scale, a = 1.0;
	for (int k = 0; k < cp->octaves; k++) {
		vec3 octoff = v3(17.13 * k, 5.31 * k, 11.71 * k);
		vec3 q = v3_add(v3_scale(p, f), octoff);
		vec3 gx, gy, gz;
		simplex3(nz, v3_add(q, off[0]), &gx);
		simplex3(nz, v3_add(q, off[1]), &gy);
		simplex3(nz, v3_add(q, off[2]), &gz);
		double m = a * f; /* chain rule: d/dp n(f p) = f * grad n */
		/* curl of potential (Px, Py, Pz) = (n0, n1, n2) */
		vec3 cc = v3(gz.y - gy.z, gx.z - gz.x, gy.x - gx.y);
		c = v3_add(c, v3_scale(cc, m));
		f *= cp->lacunarity;
		a *= cp->gain;
	}
	return c;
}
