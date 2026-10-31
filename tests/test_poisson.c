#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fluid.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

/* Test field: psi = x*y + 0.5*z (spherical harmonics l=2 and l=1)
 * Laplacian = -6xy - 1.0z */
static double psi_exact(vec3 p) { return p.x * p.y + 0.5 * p.z; }
static double lap_exact(vec3 p) { return -6 * p.x * p.y - 1.0 * p.z; }

static void lap_errors(int N, double *rms, double *mx)
{
	CubeSphere cs;
	cs_init(&cs, N);
	float *x = malloc(sizeof(float) * cs.ncells), *y = malloc(sizeof(float) * cs.ncells);
	for (int i = 0; i < cs.ncells; i++)
		x[i] = (float) psi_exact(cs_pos(&cs, i));
	fluid_laplacian(&cs, x, y);
	double s = 0, m = 0;
	for (int i = 0; i < cs.ncells; i++) {
		double e = fabs(y[i] - lap_exact(cs_pos(&cs, i)));
		s += e * e * cs.area[i];
		if (e > m) m = e;
	}
	*rms = sqrt(s / cs.total_area);
	*mx = m;
	free(x);
	free(y);
	cs_free(&cs);
}

int main(void)
{
	double prev_rms = 0;
	for (int N = 16; N <= 128; N *= 2) {
		double rms, mx;
		lap_errors(N, &rms, &mx);
		printf("Laplacian N=%4d  rms err %.3e  max err %.3e", N, rms, mx);
		if (prev_rms > 0)
			printf("  rms ratio %.2f", prev_rms / rms);
		printf("\n");
		if (N >= 64)
			CHECK(rms < 0.05, "Laplacian rms error too big at N=%d: %g", N, rms);
		prev_rms = rms;
	}

	/* Solve Laplacian(psi) = omega and compare against exact psi */
	CubeSphere cs;
	int N = 64;
	cs_init(&cs, N);
	FluidParams prm = { 0, 0, 0, 1e-6, 5000, 1 };
	Fluid F;
	fluid_init(&F, &cs, &prm);
	for (int i = 0; i < cs.ncells; i++)
		F.omega[i] = (float) lap_exact(cs_pos(&cs, i));
	fluid_remove_mean(&cs, F.omega);
	int it = fluid_solve_psi(&F);
	double s = 0, ref = 0;
	for (int i = 0; i < cs.ncells; i++) {
		double e = F.psi[i] - psi_exact(cs_pos(&cs, i));
		s += e * e * cs.area[i];
		double r = psi_exact(cs_pos(&cs, i));
		ref += r * r * cs.area[i];
	}
	double rel = sqrt(s / ref);
	printf("PCG: %d iterations, residual %.2e, psi rel L2 error %.3e\n", it, F.cg_relres, rel);
	CHECK(F.cg_relres <= 1e-6, "PCG did not converge (%g)", F.cg_relres);
	CHECK(rel < 0.02, "psi error too large: %g", rel);

	/* velocity from psi is divergence-free and its curl reproduces omega */
	fluid_compute_velocity(&F);
	float *div = malloc(sizeof(float) * cs.ncells), *om = malloc(sizeof(float) * cs.ncells);
	fluid_divergence(&cs, F.vel, div);
	fluid_curl(&cs, F.vel, om);
	double dmax = 0, crms = 0, orms = 0;
	for (int i = 0; i < cs.ncells; i++) {
		if (fabs(div[i]) > dmax) dmax = fabs(div[i]);
		double e = om[i] - F.omega[i];
		crms += e * e * cs.area[i];
		orms += (double) F.omega[i] * F.omega[i] * cs.area[i];
	}
	double crel = sqrt(crms / orms);
	printf("velocity: max |div| %.3e, curl(u) vs omega rel err %.3e\n", dmax, crel);
	CHECK(crel < 0.05, "curl of velocity doesn't match omega: %g", crel);
	free(div);
	free(om);
	fluid_free(&F);
	cs_free(&cs);

	if (failures) {
		printf("test_poisson: %d failures\n", failures);
		return 1;
	}
	printf("test_poisson: OK\n");
	return 0;
}
