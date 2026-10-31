/* Barotropic vorticity / stream-function solver on the cubed sphere.
 *
 *   d(omega)/dt + u . grad(omega) = forcing
 *   Laplacian(psi) = omega
 *   u = r_hat x grad(psi)
 */
#ifndef FLUID_H
#define FLUID_H

#include "cubesphere.h"

typedef struct {
	double forcing;   /* relaxation rate toward band vorticity (0 = off) */
	double rotation;  /* planetary rotation rate for Coriolis (0 = off) */
	int bfecc;        /* use BFECC (sharper) advection */
	double cg_tol;    /* relative residual tolerance */
	int cg_maxit;
	int multigrid;    /* use multigrid preconditioner (else Jacobi) */
} FluidParams;

typedef struct {
	CubeSphere cs;      /* coarse grid (level 0 uses the fluid's grid) */
	float *x, *b, *r, *dinv;
	/* transfer from this level to the next coarser one (bilinear prolongation
	 * P and its transpose R = P^T for restriction) */
	int *ptap;          /* [ncells*4] coarse taps for each cell of this level */
	float *pw;          /* [ncells*4] weights */
	int *rstart;        /* [ncoarse+1] CSR rows of R */
	int *ridx;          /* fine indices */
	float *rw;          /* weights */
} MGLevel;

typedef struct {
	const CubeSphere *cs;
	FluidParams prm;
	int nlev;
	MGLevel *lev;       /* lev[0] holds the fine level's scratch; lev[l>0] own a coarse grid */
	float *omega;       /* relative vorticity */
	float *omega_band;  /* forcing target (may be all zeros) */
	float *psi;         /* stream function */
	float *vel;         /* [ncells*3] tangent velocity */
	float *q0, *q1, *q2;
	float *dep_b, *dep_f; /* departure points (backward / forward) */
	float *r, *p, *Ap, *z, *dinv; /* PCG scratch */
	int cg_iters;
	double cg_relres;
} Fluid;

int fluid_init(Fluid *F, const CubeSphere *cs, const FluidParams *prm);
void fluid_free(Fluid *F);

/* Discrete operators */
void fluid_curl(const CubeSphere *cs, const float *vel, float *omega);
void fluid_divergence(const CubeSphere *cs, const float *vel, float *div);
void fluid_laplacian(const CubeSphere *cs, const float *x, float *y);
void fluid_remove_mean(const CubeSphere *cs, float *f);

/* Set the state from an initial velocity field; vel_band (may be NULL) gives
 * the forcing target. Solves for psi and velocity. */
void fluid_set_state(Fluid *F, const float *vel_initial, const float *vel_band);
int fluid_solve_psi(Fluid *F);
void fluid_compute_velocity(Fluid *F);
void fluid_step(Fluid *F, double dt);
double fluid_max_speed(const Fluid *F);

#endif
