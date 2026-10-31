/* Solid-body rotation test: a single "band" (n=1) is exactly solid-body
 * rotation about the pole. Particles must return to their start after one
 * revolution, and the vorticity field must be preserved. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fluid.h"
#include "particles.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

int main(void)
{
	CubeSphere cs;
	int N = 64;
	cs_init(&cs, N);
	int n = cs.ncells;
	double A = 1.0;

	float *vel0 = malloc(sizeof(float) * 3 * n);
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(&cs, i);
		/* A sin(colat) * east = A * (y x p) */
		v3_store(vel0 + 3 * i, v3_scale(v3_cross(v3(0, 1, 0), p), A));
	}
	FluidParams prm = { 0, 0, 1, 1e-6, 2000, 1 };
	Fluid F;
	fluid_init(&F, &cs, &prm);
	fluid_set_state(&F, vel0, NULL);

	/* reconstructed velocity matches the input; divergence ~ 0 */
	double verr = 0, vref = 0;
	for (int i = 0; i < n; i++) {
		vec3 d = v3_sub(v3_load(F.vel + 3 * i), v3_load(vel0 + 3 * i));
		verr += v3_dot(d, d) * cs.area[i];
		vref += v3_dot(v3_load(vel0 + 3 * i), v3_load(vel0 + 3 * i)) * cs.area[i];
	}
	double vrel = sqrt(verr / vref);
	float *div = malloc(sizeof(float) * n);
	fluid_divergence(&cs, F.vel, div);
	double dmax = 0;
	for (int i = 0; i < n; i++)
		if (fabs(div[i]) > dmax) dmax = fabs(div[i]);
	printf("solid body: velocity rel err %.2e, max |div| %.2e\n", vrel, dmax);
	CHECK(vrel < 0.01, "velocity reconstruction error %g", vrel);
	CHECK(dmax < 0.05, "divergence too large %g", dmax);

	/* omega should be 2*A*y */
	double oerr = 0, oref = 0;
	for (int i = 0; i < n; i++) {
		double ex = 2 * A * cs.pos[3 * i + 1];
		oerr += (F.omega[i] - ex) * (F.omega[i] - ex) * cs.area[i];
		oref += ex * ex * cs.area[i];
	}
	printf("solid body: omega vs 2Ay rel err %.2e\n", sqrt(oerr / oref));
	CHECK(sqrt(oerr / oref) < 0.01, "omega mismatch");

	float *om0 = malloc(sizeof(float) * n);
	memcpy(om0, F.omega, sizeof(float) * n);

	Particles P;
	particles_init(&P, 5000);
	float *start = malloc(sizeof(float) * 3 * P.n);
	memcpy(start, P.pos, sizeof(float) * 3 * P.n);

	double T = 2 * M_PI / A;
	int steps = 400;
	double dt = T / steps;
	for (int s = 0; s < steps; s++) {
		particles_advect(&P, &cs, F.vel, dt);
		fluid_step(&F, dt);
	}
	double pmax = 0;
	for (int i = 0; i < P.n; i++) {
		double e = v3_angle(v3_load(P.pos + 3 * i), v3_load(start + 3 * i));
		if (e > pmax) pmax = e;
	}
	printf("particles after one revolution: max position error %.2e rad (cell size %.2e)\n",
	       pmax, cs.h);
	CHECK(pmax < cs.h, "particle return error too large: %g", pmax);

	oerr = 0;
	oref = 0;
	for (int i = 0; i < n; i++) {
		oerr += (F.omega[i] - om0[i]) * (F.omega[i] - om0[i]) * cs.area[i];
		oref += om0[i] * om0[i] * cs.area[i];
	}
	printf("omega after one revolution: rel change %.2e\n", sqrt(oerr / oref));
	CHECK(sqrt(oerr / oref) < 0.02, "omega not preserved");

	free(start);
	free(om0);
	free(div);
	free(vel0);
	particles_free(&P);
	fluid_free(&F);
	cs_free(&cs);
	if (failures) {
		printf("test_flow: %d failures\n", failures);
		return 1;
	}
	printf("test_flow: OK\n");
	return 0;
}
