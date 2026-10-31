#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fluid.h"

static float *falloc(size_t n)
{
	float *f = calloc(n, sizeof(float));
	if (!f) {
		fprintf(stderr, "fluid: out of memory\n");
		exit(1);
	}
	return f;
}


/* ---------------- Multigrid preconditioner ----------------
 * Levels are cubed spheres with N, N/2, N/4, ... (rediscretized Voronoi
 * operators). Prolongation is bilinear (in equiangular coordinates, across
 * face edges too) and restriction is its transpose, and the
 * smoother is damped Jacobi with equal pre/post sweeps, so the V-cycle is a
 * symmetric linear operator suitable as a CG preconditioner. */

#ifndef MG_SWEEPS
#define MG_SWEEPS 1
#endif
#ifndef MG_OMEGA
#define MG_OMEGA 0.8
#endif
#ifndef MG_COARSE_SWEEPS
#define MG_COARSE_SWEEPS 60
#endif
#ifndef MG_CORR_SCALE
#define MG_CORR_SCALE 1.0
#endif

static inline double applyK(const CubeSphere *cs, const float *x, int i);

static void mg_init(Fluid *F)
{
	int N = F->cs->N;
	int nlev = 1;
	while ((N >> nlev) >= 4 && ((N >> (nlev - 1)) % 2) == 0)
		nlev++;
	F->nlev = nlev;
	F->lev = calloc(nlev, sizeof(MGLevel));
	F->lev[0].dinv = F->dinv;
	for (int l = 1; l < nlev; l++) {
		MGLevel *L = &F->lev[l];
		cs_init(&L->cs, N >> l);
		int n = L->cs.ncells;
		L->x = falloc(n);
		L->b = falloc(n);
		L->r = falloc(n);
		L->dinv = falloc(n);
		for (int i = 0; i < n; i++) {
			double s = 0;
			for (int k = 0; k < L->cs.nnb[i]; k++)
				s += L->cs.w[CS_MAXNB * i + k];
			L->dinv[i] = (float) (1.0 / s);
		}
		/* transfer operators between level l-1 (fine) and l (coarse) */
		const CubeSphere *fine = l == 1 ? F->cs : &F->lev[l - 1].cs;
		MGLevel *Lf = &F->lev[l - 1];
		int nf = fine->ncells, nc = n;
		Lf->ptap = malloc(sizeof(int) * 4 * nf);
		Lf->pw = malloc(sizeof(float) * 4 * nf);
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < nf; i++) {
			int idx[4];
			double wt[4];
			cs_taps(&L->cs, cs_pos(fine, i), idx, wt);
			for (int k = 0; k < 4; k++) {
				Lf->ptap[4 * i + k] = idx[k];
				Lf->pw[4 * i + k] = (float) wt[k];
			}
		}
		Lf->rstart = calloc(nc + 1, sizeof(int));
		for (int i = 0; i < 4 * nf; i++)
			Lf->rstart[Lf->ptap[i] + 1]++;
		for (int c = 0; c < nc; c++)
			Lf->rstart[c + 1] += Lf->rstart[c];
		Lf->ridx = malloc(sizeof(int) * 4 * nf);
		Lf->rw = malloc(sizeof(float) * 4 * nf);
		int *cur = malloc(sizeof(int) * nc);
		memcpy(cur, Lf->rstart, sizeof(int) * nc);
		for (int i = 0; i < nf; i++)
			for (int k = 0; k < 4; k++) {
				int c = Lf->ptap[4 * i + k];
				Lf->ridx[cur[c]] = i;
				Lf->rw[cur[c]] = Lf->pw[4 * i + k];
				cur[c]++;
			}
		free(cur);
	}
}

static void mg_smooth(const CubeSphere *cs, const float *dinv, const float *b, float *x, float *r, int sweeps)
{
	int n = cs->ncells;
	for (int s = 0; s < sweeps; s++) {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)
			r[i] = (float) (b[i] - applyK(cs, x, i));
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)
			x[i] += (float) (MG_OMEGA * dinv[i] * r[i]);
	}
}

/* x = V(b) at level l; x is overwritten. */
static void mg_vcycle(Fluid *F, int l, const float *b, float *x, float *r)
{
	const CubeSphere *cs = l == 0 ? F->cs : &F->lev[l].cs;
	const float *dinv = F->lev[l].dinv;
	int n = cs->ncells;
	memset(x, 0, sizeof(float) * n);
	if (l == F->nlev - 1) {
		mg_smooth(cs, dinv, b, x, r, MG_COARSE_SWEEPS);
		return;
	}
	mg_smooth(cs, dinv, b, x, r, MG_SWEEPS);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		r[i] = (float) (b[i] - applyK(cs, x, i));
	MGLevel *C = &F->lev[l + 1];
	const MGLevel *L = &F->lev[l];
	#pragma omp parallel for schedule(static)
	for (int c = 0; c < C->cs.ncells; c++) {
		double acc = 0;
		for (int k = L->rstart[c]; k < L->rstart[c + 1]; k++)
			acc += L->rw[k] * r[L->ridx[k]];
		C->b[c] = (float) acc;
	}
	mg_vcycle(F, l + 1, C->b, C->x, C->r);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		const int *t = L->ptap + 4 * i;
		const float *w = L->pw + 4 * i;
		x[i] += (float) (MG_CORR_SCALE * (w[0] * C->x[t[0]] + w[1] * C->x[t[1]] +
						   w[2] * C->x[t[2]] + w[3] * C->x[t[3]]));
	}
	mg_smooth(cs, dinv, b, x, r, MG_SWEEPS);
}

/* z = M^-1 r */
static void precondition(Fluid *F, const float *r, float *z)
{
	int n = F->cs->ncells;
	if (F->nlev > 0) {
		/* lev[0] scratch for residual: reuse Ap */
		mg_vcycle(F, 0, r, z, F->Ap);
	} else {
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)
			z[i] = F->dinv[i] * r[i];
	}
}

int fluid_init(Fluid *F, const CubeSphere *cs, const FluidParams *prm)
{
	memset(F, 0, sizeof(*F));
	F->cs = cs;
	F->prm = *prm;
	size_t n = cs->ncells;
	F->omega = falloc(n);
	F->omega_band = falloc(n);
	F->psi = falloc(n);
	F->vel = falloc(3 * n);
	F->q0 = falloc(n);
	F->q1 = falloc(n);
	F->q2 = falloc(n);
	F->dep_b = falloc(3 * n);
	F->dep_f = falloc(3 * n);
	F->r = falloc(n);
	F->p = falloc(n);
	F->Ap = falloc(n);
	F->z = falloc(n);
	F->dinv = falloc(n);
	for (size_t i = 0; i < n; i++) {
		double s = 0;
		for (int k = 0; k < cs->nnb[i]; k++)
			s += cs->w[CS_MAXNB * i + k];
		F->dinv[i] = (float) (1.0 / s);
	}
	if (prm->multigrid)
		mg_init(F);
	return 0;
}

void fluid_free(Fluid *F)
{
	free(F->omega); free(F->omega_band); free(F->psi); free(F->vel);
	free(F->q0); free(F->q1); free(F->q2);
	free(F->dep_b); free(F->dep_f);
	free(F->r); free(F->p); free(F->Ap); free(F->z); free(F->dinv);
	for (int l = 0; l < F->nlev; l++) {
		MGLevel *L = &F->lev[l];
		if (l > 0) {
			cs_free(&L->cs);
			free(L->x); free(L->b); free(L->r); free(L->dinv);
		}
		free(L->ptap); free(L->pw); free(L->rstart); free(L->ridx); free(L->rw);
	}
	free(F->lev);
	memset(F, 0, sizeof(*F));
}

/* Divergence by Voronoi finite volumes: flux through each Voronoi edge uses
 * the mean of the two cell values dotted with the chord direction, which is
 * exactly normal to the (great-circle) Voronoi edge. Conservative. */
void fluid_divergence(const CubeSphere *cs, const float *vel, float *div)
{
	int n = cs->ncells;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(cs, i);
		vec3 ui = v3_load(vel + 3 * i);
		double s = 0;
		for (int k = 0; k < cs->nnb[i]; k++) {
			int j = cs->nbr[CS_MAXNB * i + k];
			vec3 nrm = v3_normalize(v3_sub(cs_pos(cs, j), p));
			vec3 um = v3_add(ui, v3_load(vel + 3 * j));
			s += cs->elen[CS_MAXNB * i + k] * 0.5 * v3_dot(um, nrm);
		}
		div[i] = (float) (s / cs->area[i]);
	}
}

/* Radial vorticity: r . curl(u) = div(u x r) for tangent fields. */
void fluid_curl(const CubeSphere *cs, const float *vel, float *omega)
{
	int n = cs->ncells;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(cs, i);
		vec3 wi = v3_cross(v3_load(vel + 3 * i), p);
		double s = 0;
		for (int k = 0; k < cs->nnb[i]; k++) {
			int j = cs->nbr[CS_MAXNB * i + k];
			vec3 pj = cs_pos(cs, j);
			vec3 wj = v3_cross(v3_load(vel + 3 * j), pj);
			vec3 nrm = v3_normalize(v3_sub(pj, p));
			s += cs->elen[CS_MAXNB * i + k] * 0.5 * v3_dot(v3_add(wi, wj), nrm);
		}
		omega[i] = (float) (s / cs->area[i]);
	}
}

void fluid_laplacian(const CubeSphere *cs, const float *x, float *y)
{
	int n = cs->ncells;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		double s = 0, xi = x[i];
		for (int k = 0; k < cs->nnb[i]; k++)
			s += cs->w[CS_MAXNB * i + k] * (x[cs->nbr[CS_MAXNB * i + k]] - xi);
		y[i] = (float) (s / cs->area[i]);
	}
}

void fluid_remove_mean(const CubeSphere *cs, float *f)
{
	int n = cs->ncells;
	double s = 0;
	#pragma omp parallel for schedule(static) reduction(+:s)
	for (int i = 0; i < n; i++)
		s += (double) cs->area[i] * f[i];
	float m = (float) (s / cs->total_area);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		f[i] -= m;
}

/* K x = sum_j w_ij (x_i - x_j)  (symmetric positive semi-definite) */
static inline double applyK(const CubeSphere *cs, const float *x, int i)
{
	const int *nb = cs->nbr + CS_MAXNB * i;
	const float *w = cs->w + CS_MAXNB * i;
	double xi = x[i], s = 0;
	for (int k = 0; k < cs->nnb[i]; k++)
		s += w[k] * (xi - x[nb[k]]);
	return s;
}

/* Solve Laplacian(psi) = omega, i.e. K psi = -A omega, with preconditioned
 * CG, warm-started from the current psi. Returns iteration count. */
int fluid_solve_psi(Fluid *F)
{
	const CubeSphere *cs = F->cs;
	int n = cs->ncells;
	float *psi = F->psi, *r = F->r, *p = F->p, *Ap = F->Ap, *z = F->z;
	double bb = 0, rr = 0, rz = 0;

	#pragma omp parallel for schedule(static) reduction(+:bb,rr)
	for (int i = 0; i < n; i++) {
		double b = -(double) cs->area[i] * F->omega[i];
		double ri = b - applyK(cs, psi, i);
		r[i] = (float) ri;
		bb += b * b;
		rr += ri * ri;
	}
	F->cg_iters = 0;
	if (bb == 0.0) {
		memset(psi, 0, sizeof(float) * n);
		F->cg_relres = 0;
		return 0;
	}
	double tol2 = F->prm.cg_tol * F->prm.cg_tol * bb;
	int it = 0;
	if (rr > tol2) {
		precondition(F, r, z);
		#pragma omp parallel for schedule(static) reduction(+:rz)
		for (int i = 0; i < n; i++) {
			p[i] = z[i];
			rz += (double) r[i] * z[i];
		}
	}
	while (rr > tol2 && it < F->prm.cg_maxit) {
		double pAp = 0;
		#pragma omp parallel for schedule(static) reduction(+:pAp)
		for (int i = 0; i < n; i++) {
			double a = applyK(cs, p, i);
			Ap[i] = (float) a;
			pAp += a * p[i];
		}
		if (pAp <= 0)
			break;
		double alpha = rz / pAp;
		rr = 0;
		#pragma omp parallel for schedule(static) reduction(+:rr)
		for (int i = 0; i < n; i++) {
			psi[i] += (float) (alpha * p[i]);
			double ri = r[i] - alpha * Ap[i];
			r[i] = (float) ri;
			rr += ri * ri;
		}
		it++;
		if (rr <= tol2)
			break;
		precondition(F, r, z);
		double rznew = 0;
		#pragma omp parallel for schedule(static) reduction(+:rznew)
		for (int i = 0; i < n; i++)
			rznew += (double) r[i] * z[i];
		double beta = rznew / rz;
		rz = rznew;
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)
			p[i] = (float) (z[i] + beta * p[i]);
	}
	fluid_remove_mean(cs, psi);
	F->cg_iters = it;
	F->cg_relres = sqrt(rr / bb);
	return it;
}

/* u = p x grad(psi); gradient via least-squares coefficients. */
void fluid_compute_velocity(Fluid *F)
{
	const CubeSphere *cs = F->cs;
	int n = cs->ncells;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(cs, i);
		vec3 g = v3(0, 0, 0);
		double pi = F->psi[i];
		for (int k = 0; k < cs->nnb[i]; k++) {
			int j = cs->nbr[CS_MAXNB * i + k];
			g = v3_add(g, v3_scale(v3_load(cs->gc + 3 * (CS_MAXNB * i + k)), F->psi[j] - pi));
		}
		v3_store(F->vel + 3 * i, v3_cross(p, g));
	}
}

void fluid_set_state(Fluid *F, const float *vel_initial, const float *vel_band)
{
	const CubeSphere *cs = F->cs;
	fluid_curl(cs, vel_initial, F->omega);
	fluid_remove_mean(cs, F->omega);
	if (vel_band) {
		fluid_curl(cs, vel_band, F->omega_band);
		fluid_remove_mean(cs, F->omega_band);
	} else {
		memset(F->omega_band, 0, sizeof(float) * cs->ncells);
	}
	memset(F->psi, 0, sizeof(float) * cs->ncells);
	int save = F->prm.cg_maxit;
	F->prm.cg_maxit = save * 20 > 20000 ? save * 20 : 20000; /* cold start */
	fluid_solve_psi(F);
	F->prm.cg_maxit = save;
	fluid_compute_velocity(F);
}

static void compute_departures(Fluid *F, double dt, int forward_too)
{
	const CubeSphere *cs = F->cs;
	int n = cs->ncells;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(cs, i);
		vec3 u = v3_load(F->vel + 3 * i);
		vec3 pm = sphere_move(p, u, -0.5 * dt);
		vec3 um = v3_tangent(cs_sample_vec(cs, F->vel, pm), p);
		v3_store(F->dep_b + 3 * i, sphere_move(p, um, -dt));
		if (forward_too) {
			pm = sphere_move(p, u, 0.5 * dt);
			um = v3_tangent(cs_sample_vec(cs, F->vel, pm), p);
			v3_store(F->dep_f + 3 * i, sphere_move(p, um, dt));
		}
	}
}

static void advect(const CubeSphere *cs, const float *src, const float *dep, float *dst)
{
	int n = cs->ncells;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		dst[i] = (float) cs_sample_scalar(cs, src, v3_load(dep + 3 * i));
}

void fluid_step(Fluid *F, double dt)
{
	const CubeSphere *cs = F->cs;
	int n = cs->ncells;
	double twoOmega = 2.0 * F->prm.rotation;
	int bfecc = F->prm.bfecc;

	compute_departures(F, dt, bfecc);

	/* advected quantity: absolute vorticity q = omega + 2*Omega*y */
	float *q = F->q0;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		q[i] = (float) (F->omega[i] + twoOmega * cs->pos[3 * i + 1]);

	float *qn = F->q1;
	if (!bfecc) {
		advect(cs, q, F->dep_b, qn);
	} else {
		float *q1 = F->q1, *q2 = F->q2;
		advect(cs, q, F->dep_b, q1);           /* forward */
		advect(cs, q1, F->dep_f, q2);          /* backward */
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++)            /* corrected source */
			q2[i] = q[i] + 0.5f * (q[i] - q2[i]);
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; i++) {
			int idx[4];
			double wt[4];
			cs_taps(cs, v3_load(F->dep_b + 3 * i), idx, wt);
			double v = 0, lo = 1e30, hi = -1e30;
			for (int k = 0; k < 4; k++) {
				v += wt[k] * q2[idx[k]];
				double o = q[idx[k]];
				if (o < lo) lo = o;
				if (o > hi) hi = o;
			}
			if (v < lo) v = lo;   /* limiter: no new extrema */
			if (v > hi) v = hi;
			q1[i] = (float) v;
		}
		qn = q1;
	}

	double relax = F->prm.forcing > 0 ? 1.0 - exp(-F->prm.forcing * dt) : 0.0;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		double w = qn[i] - twoOmega * cs->pos[3 * i + 1];
		w += relax * (F->omega_band[i] - w);
		F->omega[i] = (float) w;
	}
	fluid_remove_mean(cs, F->omega);
	fluid_solve_psi(F);
	fluid_compute_velocity(F);
}

double fluid_max_speed(const Fluid *F)
{
	int n = F->cs->ncells;
	double m = 0;
	#pragma omp parallel for schedule(static) reduction(max:m)
	for (int i = 0; i < n; i++) {
		const float *v = F->vel + 3 * i;
		double s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
		if (s > m) m = s;
	}
	return sqrt(m);
}
