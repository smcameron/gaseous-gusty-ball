#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "noise.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static double frand(unsigned *s)
{
	*s = *s * 1664525u + 1013904223u;
	return (*s >> 8) / 16777216.0;
}

/* Mean spacing of sign changes of the east velocity component along the
 * equator, a proxy for whorl size. */
static double whorl_size(const Noise *nz, double scale)
{
	CurlParams cp = { scale, 2.0, 0.5, 1 };
	int M = 20000, changes = 0;
	double prev = 0;
	for (int k = 0; k <= M; k++) {
		double lon = 2 * M_PI * k / M;
		vec3 p = lonlat_to_dir(lon, 0.0);
		vec3 east = v3(cos(lon), 0, -sin(lon));
		vec3 v = v3_tangent(curl_noise(nz, p, &cp), p);
		double c = v3_dot(v, v3_cross(p, east)); /* north component */
		if (k > 0 && (c > 0) != (prev > 0))
			changes++;
		prev = c;
	}
	return changes ? 2 * M_PI / changes : 1e9;
}

int main(void)
{
	Noise nz;
	noise_init(&nz, 42);
	unsigned s = 7;

	/* analytic gradient vs finite differences */
	double maxrel = 0;
	for (int k = 0; k < 2000; k++) {
		vec3 p = v3(frand(&s) * 20 - 10, frand(&s) * 20 - 10, frand(&s) * 20 - 10);
		vec3 g;
		simplex3(&nz, p, &g);
		double e = 1e-5;
		vec3 fd = v3(
			(simplex3(&nz, v3(p.x + e, p.y, p.z), NULL) - simplex3(&nz, v3(p.x - e, p.y, p.z), NULL)) / (2 * e),
			(simplex3(&nz, v3(p.x, p.y + e, p.z), NULL) - simplex3(&nz, v3(p.x, p.y - e, p.z), NULL)) / (2 * e),
			(simplex3(&nz, v3(p.x, p.y, p.z + e), NULL) - simplex3(&nz, v3(p.x, p.y, p.z - e), NULL)) / (2 * e));
		double err = v3_len(v3_sub(g, fd)) / (1.0 + v3_len(fd));
		if (err > maxrel) maxrel = err;
	}
	printf("simplex gradient vs FD: max rel err %.2e\n", maxrel);
	CHECK(maxrel < 1e-4, "gradient mismatch %g", maxrel);

	/* 3D curl noise is divergence-free; projected field is tangent */
	CurlParams cp = { 3.0, 2.0, 0.5, 4 };
	double maxdiv = 0, maxmag = 0, maxdot = 0;
	for (int k = 0; k < 2000; k++) {
		vec3 p = v3_normalize(v3(frand(&s) * 2 - 1, frand(&s) * 2 - 1, frand(&s) * 2 - 1));
		double e = 1e-5;
		vec3 c = curl_noise(&nz, p, &cp);
		double div =
			(curl_noise(&nz, v3(p.x + e, p.y, p.z), &cp).x - curl_noise(&nz, v3(p.x - e, p.y, p.z), &cp).x +
			 curl_noise(&nz, v3(p.x, p.y + e, p.z), &cp).y - curl_noise(&nz, v3(p.x, p.y - e, p.z), &cp).y +
			 curl_noise(&nz, v3(p.x, p.y, p.z + e), &cp).z - curl_noise(&nz, v3(p.x, p.y, p.z - e), &cp).z) / (2 * e);
		if (fabs(div) > maxdiv) maxdiv = fabs(div);
		if (v3_len(c) > maxmag) maxmag = v3_len(c);
		vec3 t = v3_tangent(c, p);
		if (fabs(v3_dot(t, p)) > maxdot) maxdot = fabs(v3_dot(t, p));
	}
	printf("curl noise: max |div| %.2e (max |v| %.2f), max |v.p| after projection %.1e\n",
	       maxdiv, maxmag, maxdot);
	CHECK(maxdiv < 1e-3 * maxmag, "curl noise not divergence-free: %g", maxdiv);
	CHECK(maxdot < 1e-12, "projected field not tangent");

	/* doubling scale halves whorl size */
	double w1 = whorl_size(&nz, 3.0), w2 = whorl_size(&nz, 6.0), w3 = whorl_size(&nz, 12.0);
	printf("whorl size: scale 3 -> %.3f, 6 -> %.3f, 12 -> %.3f (ratios %.2f, %.2f)\n",
	       w1, w2, w3, w1 / w2, w2 / w3);
	CHECK(w1 / w2 > 1.5 && w1 / w2 < 2.7, "scale ratio 3->6 = %g", w1 / w2);
	CHECK(w2 / w3 > 1.5 && w2 / w3 < 2.7, "scale ratio 6->12 = %g", w2 / w3);

	if (failures) {
		printf("test_noise: %d failures\n", failures);
		return 1;
	}
	printf("test_noise: OK\n");
	return 0;
}
