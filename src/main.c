/* gaseous-gusty-ball: fluid simulation on a sphere rendered as particle trails into
 * cubemap images, for making gas giant planet textures. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <getopt.h>
#include <sys/stat.h>
#include <omp.h>

#include "cubesphere.h"
#include "noise.h"
#include "fluid.h"
#include "particles.h"
#include "render.h"
#include "image_io.h"

typedef struct {
	const char *input;
	const char *outdir;
	const char *prefix;
	int particles, grid, res;
	int bands;
	double band_amp;
	double noise_amp, noise_scale, noise_lacunarity, noise_gain;
	int noise_octaves;
	double forcing, rotation;
	double dt, cfl;
	int steps, every;
	int sequence, net, preview, gl_layout;
	double fade, trail_alpha, splat_radius;
	int transparent;
	float bg[3];
	unsigned int seed;
	int threads;
	int bfecc;
	double cg_tol;
	int cg_maxit;
} Opts;

static void usage(void)
{
	fprintf(stderr,
"usage: gaseous-gusty-ball -i input.png [options]\n"
"\n"
"Simulates fluid flow on a sphere, advects colored particles (colors sampled\n"
"from the input image, equirectangular mapping) and renders fading particle\n"
"trails into 6 cubemap face images.\n"
"\n"
"Input / output:\n"
"  -i, --input FILE          input PNG (width wraps longitude, height pole to pole)\n"
"  -o, --outdir DIR          output directory (default: out)\n"
"      --prefix P            output file prefix (default: gg)\n"
"      --layout cross|gl     cross: P0..P5.png faces as in the cross net (default)\n"
"                            gl: P-posx.png .. P-negz.png (OpenGL cubemap order)\n"
"      --sequence            write numbered frames instead of overwriting\n"
"      --net                 also write the assembled 4x3 cross net image\n"
"      --preview             also write an equirectangular preview image\n"
"      --res R               output face resolution (default: 1024)\n"
"      --bg transparent|RRGGBB  background behind trails (default: transparent)\n"
"\n"
"Simulation:\n"
"      --particles M         number of particles (default: 1000000)\n"
"      --grid N              fluid cells per cube face edge (default: 256)\n"
"      --bands n             number of counter-rotating zonal bands (default: 8)\n"
"      --band-amp A          band speed amplitude (default: 1.0)\n"
"      --noise-amp A         curl noise RMS speed (default: 0.5)\n"
"      --noise-scale s       curl noise base frequency; small = big whorls (default: 3.0)\n"
"      --noise-octaves k     curl noise octaves (default: 4)\n"
"      --noise-lacunarity l  frequency ratio between octaves (default: 2.0)\n"
"      --noise-gain g        amplitude ratio between octaves (default: 0.5)\n"
"      --forcing k           relaxation rate toward band flow, 0 = off (default: 0)\n"
"      --rotation W          planetary rotation rate (Coriolis), 0 = off (default: 0)\n"
"      --no-bfecc            use plain (more diffusive) semi-Lagrangian advection\n"
"      --dt T                fixed time step (default: automatic from --cfl)\n"
"      --cfl C               CFL number for automatic time step (default: 1.0)\n"
"      --steps S             number of steps (default: 2000)\n"
"      --every K             output every K steps (default: 50)\n"
"      --seed S              random seed (default: 1)\n"
"      --threads T           OpenMP threads (default: all)\n"
"\n"
"Trails:\n"
"      --fade f              fraction of trail opacity removed per step (default: 0.02)\n"
"      --trail-alpha a       opacity of each trail deposit (default: 0.35)\n"
"      --splat-radius r      trail half-width in texels (default: 1.0)\n");
}

enum {
	O_PREFIX = 256, O_LAYOUT, O_SEQUENCE, O_NET, O_PREVIEW, O_RES, O_BG,
	O_PARTICLES, O_GRID, O_BANDS, O_BANDAMP, O_NAMP, O_NSCALE, O_NOCT,
	O_NLAC, O_NGAIN, O_FORCING, O_ROTATION, O_NOBFECC, O_DT, O_CFL,
	O_STEPS, O_EVERY, O_SEED, O_THREADS, O_FADE, O_TALPHA, O_SPLATR,
	O_CGTOL, O_CGMAXIT,
};

static double parse_d(const char *s, const char *name)
{
	char *e;
	errno = 0;
	double v = strtod(s, &e);
	if (errno || *e || e == s) {
		fprintf(stderr, "bad value for %s: %s\n", name, s);
		exit(1);
	}
	return v;
}

static int parse_i(const char *s, const char *name)
{
	char *e;
	errno = 0;
	long v = strtol(s, &e, 10);
	if (errno || *e || e == s) {
		fprintf(stderr, "bad value for %s: %s\n", name, s);
		exit(1);
	}
	return (int) v;
}

static void parse_args(int argc, char **argv, Opts *o)
{
	static struct option lo[] = {
		{"input", required_argument, 0, 'i'},
		{"outdir", required_argument, 0, 'o'},
		{"help", no_argument, 0, 'h'},
		{"prefix", required_argument, 0, O_PREFIX},
		{"layout", required_argument, 0, O_LAYOUT},
		{"sequence", no_argument, 0, O_SEQUENCE},
		{"net", no_argument, 0, O_NET},
		{"preview", no_argument, 0, O_PREVIEW},
		{"res", required_argument, 0, O_RES},
		{"bg", required_argument, 0, O_BG},
		{"particles", required_argument, 0, O_PARTICLES},
		{"grid", required_argument, 0, O_GRID},
		{"bands", required_argument, 0, O_BANDS},
		{"band-amp", required_argument, 0, O_BANDAMP},
		{"noise-amp", required_argument, 0, O_NAMP},
		{"noise-scale", required_argument, 0, O_NSCALE},
		{"noise-octaves", required_argument, 0, O_NOCT},
		{"noise-lacunarity", required_argument, 0, O_NLAC},
		{"noise-gain", required_argument, 0, O_NGAIN},
		{"forcing", required_argument, 0, O_FORCING},
		{"rotation", required_argument, 0, O_ROTATION},
		{"no-bfecc", no_argument, 0, O_NOBFECC},
		{"dt", required_argument, 0, O_DT},
		{"cfl", required_argument, 0, O_CFL},
		{"steps", required_argument, 0, O_STEPS},
		{"every", required_argument, 0, O_EVERY},
		{"seed", required_argument, 0, O_SEED},
		{"threads", required_argument, 0, O_THREADS},
		{"fade", required_argument, 0, O_FADE},
		{"trail-alpha", required_argument, 0, O_TALPHA},
		{"splat-radius", required_argument, 0, O_SPLATR},
		{"cg-tol", required_argument, 0, O_CGTOL},
		{"cg-maxit", required_argument, 0, O_CGMAXIT},
		{0, 0, 0, 0},
	};
	int c;
	while ((c = getopt_long(argc, argv, "i:o:h", lo, NULL)) != -1) {
		switch (c) {
		case 'i': o->input = optarg; break;
		case 'o': o->outdir = optarg; break;
		case 'h': usage(); exit(0);
		case O_PREFIX: o->prefix = optarg; break;
		case O_LAYOUT:
			if (!strcmp(optarg, "cross")) o->gl_layout = 0;
			else if (!strcmp(optarg, "gl")) o->gl_layout = 1;
			else { fprintf(stderr, "bad --layout: %s\n", optarg); exit(1); }
			break;
		case O_SEQUENCE: o->sequence = 1; break;
		case O_NET: o->net = 1; break;
		case O_PREVIEW: o->preview = 1; break;
		case O_RES: o->res = parse_i(optarg, "--res"); break;
		case O_BG:
			if (!strcmp(optarg, "transparent")) {
				o->transparent = 1;
			} else {
				unsigned int rgb;
				const char *s = optarg[0] == '#' ? optarg + 1 : optarg;
				if (strlen(s) != 6 || sscanf(s, "%x", &rgb) != 1) {
					fprintf(stderr, "bad --bg: %s\n", optarg);
					exit(1);
				}
				o->transparent = 0;
				o->bg[0] = ((rgb >> 16) & 255) / 255.0f;
				o->bg[1] = ((rgb >> 8) & 255) / 255.0f;
				o->bg[2] = (rgb & 255) / 255.0f;
			}
			break;
		case O_PARTICLES: o->particles = parse_i(optarg, "--particles"); break;
		case O_GRID: o->grid = parse_i(optarg, "--grid"); break;
		case O_BANDS: o->bands = parse_i(optarg, "--bands"); break;
		case O_BANDAMP: o->band_amp = parse_d(optarg, "--band-amp"); break;
		case O_NAMP: o->noise_amp = parse_d(optarg, "--noise-amp"); break;
		case O_NSCALE: o->noise_scale = parse_d(optarg, "--noise-scale"); break;
		case O_NOCT: o->noise_octaves = parse_i(optarg, "--noise-octaves"); break;
		case O_NLAC: o->noise_lacunarity = parse_d(optarg, "--noise-lacunarity"); break;
		case O_NGAIN: o->noise_gain = parse_d(optarg, "--noise-gain"); break;
		case O_FORCING: o->forcing = parse_d(optarg, "--forcing"); break;
		case O_ROTATION: o->rotation = parse_d(optarg, "--rotation"); break;
		case O_NOBFECC: o->bfecc = 0; break;
		case O_DT: o->dt = parse_d(optarg, "--dt"); break;
		case O_CFL: o->cfl = parse_d(optarg, "--cfl"); break;
		case O_STEPS: o->steps = parse_i(optarg, "--steps"); break;
		case O_EVERY: o->every = parse_i(optarg, "--every"); break;
		case O_SEED: o->seed = (unsigned int) parse_i(optarg, "--seed"); break;
		case O_THREADS: o->threads = parse_i(optarg, "--threads"); break;
		case O_FADE: o->fade = parse_d(optarg, "--fade"); break;
		case O_TALPHA: o->trail_alpha = parse_d(optarg, "--trail-alpha"); break;
		case O_SPLATR: o->splat_radius = parse_d(optarg, "--splat-radius"); break;
		case O_CGTOL: o->cg_tol = parse_d(optarg, "--cg-tol"); break;
		case O_CGMAXIT: o->cg_maxit = parse_i(optarg, "--cg-maxit"); break;
		default: usage(); exit(1);
		}
	}
	if (!o->input) {
		usage();
		exit(1);
	}
	if (o->particles < 1 || o->grid < 4 || o->res < 4 || o->bands < 0 || o->steps < 0 ||
	    o->every < 1 || o->noise_octaves < 0 || o->fade < 0 || o->fade >= 1 ||
	    o->trail_alpha <= 0 || o->trail_alpha > 1) {
		fprintf(stderr, "invalid option value\n");
		exit(1);
	}
}

static int mkdir_p(const char *path)
{
	char buf[4096];
	snprintf(buf, sizeof(buf), "%s", path);
	for (char *p = buf + 1; *p; p++) {
		if (*p == '/') {
			*p = 0;
			if (mkdir(buf, 0777) && errno != EEXIST)
				return -1;
			*p = '/';
		}
	}
	if (mkdir(buf, 0777) && errno != EEXIST)
		return -1;
	return 0;
}

static const char *gl_names[6] = { "posx", "negx", "posy", "negy", "posz", "negz" };
static const int gl_to_face[6] = { 1, 3, 4, 5, 0, 2 }; /* GL face order -> internal face */

static void write_outputs(const Opts *o, const Render *r, int step)
{
	int R = r->R;
	const float *bg = o->transparent ? NULL : o->bg;
	int ch = o->transparent ? 4 : 3;
	char seq[32] = "";
	if (o->sequence)
		snprintf(seq, sizeof(seq), "-%05d", step);

	#pragma omp parallel for schedule(dynamic, 1)
	for (int k = 0; k < 6; k++) {
		int face = o->gl_layout ? gl_to_face[k] : k;
		char path[4096];
		if (o->gl_layout)
			snprintf(path, sizeof(path), "%s/%s%s-%s.png", o->outdir, o->prefix, seq, gl_names[k]);
		else
			snprintf(path, sizeof(path), "%s/%s%s%s%d.png", o->outdir, o->prefix, seq,
				 o->sequence ? "-" : "", k);
		unsigned char *img = malloc((size_t) R * R * ch);
		for (int y = 0; y < R; y++)
			for (int x = 0; x < R; x++) {
				float c[4];
				render_texel(r, face, x, y, c);
				render_pixel8(c, bg, img + ((size_t) y * R + x) * ch);
			}
		png_save(path, R, R, ch, img);
		free(img);
	}

	if (o->net) {
		int W = 4 * R, H = 3 * R;
		unsigned char *img = calloc((size_t) W * H, 4);
		static const int cell_x[6] = { 0, 1, 2, 3, 0, 0 }, cell_y[6] = { 1, 1, 1, 1, 0, 2 };
		#pragma omp parallel for schedule(static)
		for (int fy = 0; fy < 6 * R; fy++) {
			int f = fy / R, y = fy % R;
			for (int x = 0; x < R; x++) {
				float c[4];
				unsigned char px[4] = { 0, 0, 0, 255 };
				render_texel(r, f, x, y, c);
				render_pixel8(c, bg, px);
				memcpy(img + (((size_t) cell_y[f] * R + y) * W + cell_x[f] * R + x) * 4, px, 4);
			}
		}
		char path[4096];
		snprintf(path, sizeof(path), "%s/%s%s-net.png", o->outdir, o->prefix, seq);
		png_save(path, W, H, 4, img);
		free(img);
	}

	if (o->preview) {
		int W = 2 * R, H = R;
		unsigned char *img = malloc((size_t) W * H * ch);
		#pragma omp parallel for schedule(static)
		for (int y = 0; y < H; y++) {
			double lat = M_PI / 2 - (y + 0.5) / H * M_PI;
			for (int x = 0; x < W; x++) {
				double lon = (x + 0.5) / W * 2 * M_PI - M_PI;
				vec3 d = lonlat_to_dir(lon, lat);
				float c[4];
				render_sample_dir(r, d.x, d.y, d.z, c);
				render_pixel8(c, bg, img + ((size_t) y * W + x) * ch);
			}
		}
		char path[4096];
		snprintf(path, sizeof(path), "%s/%s%s-preview.png", o->outdir, o->prefix, seq);
		png_save(path, W, H, ch, img);
		free(img);
	}
}

/* Initial velocity fields: curl noise (projected, RMS-normalized) + zonal bands. */
static void initial_velocity(const Opts *o, const CubeSphere *cs, float *vel, float *vband)
{
	int n = cs->ncells;
	Noise nz;
	noise_init(&nz, o->seed);
	CurlParams cp = { o->noise_scale, o->noise_lacunarity, o->noise_gain, o->noise_octaves };

	double ss = 0;
	#pragma omp parallel for schedule(static) reduction(+:ss)
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(cs, i);
		vec3 v = o->noise_octaves > 0 ? v3_tangent(curl_noise(&nz, p, &cp), p) : v3(0, 0, 0);
		v3_store(vel + 3 * i, v);
		ss += v3_dot(v, v) * cs->area[i];
	}
	double rms = sqrt(ss / cs->total_area);
	double sc = rms > 0 ? o->noise_amp / rms : 0;

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		vec3 p = cs_pos(cs, i);
		/* u_east = A sin(n * colatitude); (y x p) has magnitude sin(colatitude) */
		double colat = acos(p.y > 1 ? 1 : (p.y < -1 ? -1 : p.y));
		double s = sin(colat);
		double mag = o->band_amp * sin(o->bands * colat);
		vec3 east = v3_cross(v3(0, 1, 0), p);
		vec3 b = s > 1e-9 ? v3_scale(east, mag / s) : v3(0, 0, 0);
		v3_store(vband + 3 * i, b);
		vec3 v = v3_add(v3_scale(v3_load(vel + 3 * i), sc), b);
		v3_store(vel + 3 * i, v);
	}

	if (o->noise_octaves > 0) {
		double fmax = o->noise_scale * pow(o->noise_lacunarity, o->noise_octaves - 1);
		if (1.0 / fmax < 4 * cs->h)
			fprintf(stderr, "warning: finest noise octave (feature size %.4f) is smaller than "
				"~4 fluid cells (%.4f); increase --grid or reduce --noise-octaves\n",
				1.0 / fmax, 4 * cs->h);
	}
}

int main(int argc, char **argv)
{
	Opts o = {
		.outdir = "out", .prefix = "gg",
		.particles = 1000000, .grid = 256, .res = 1024,
		.bands = 8, .band_amp = 1.0,
		.noise_amp = 0.5, .noise_scale = 3.0, .noise_lacunarity = 2.0, .noise_gain = 0.5,
		.noise_octaves = 4,
		.cfl = 1.0, .steps = 2000, .every = 50,
		.fade = 0.02, .trail_alpha = 0.35, .splat_radius = 1.0,
		.transparent = 1, .seed = 1, .bfecc = 1,
		.cg_tol = 1e-4, .cg_maxit = 500,
	};
	parse_args(argc, argv, &o);
	if (o.threads > 0)
		omp_set_num_threads(o.threads);
	if (mkdir_p(o.outdir)) {
		fprintf(stderr, "cannot create %s: %s\n", o.outdir, strerror(errno));
		return 1;
	}

	double t0 = omp_get_wtime();
	int iw, ih;
	float *img = png_load_rgba(o.input, &iw, &ih);
	if (!img)
		return 1;

	CubeSphere cs;
	if (cs_init(&cs, o.grid))
		return 1;
	fprintf(stderr, "grid: 6x%dx%d cells (%d), %d threads\n", o.grid, o.grid, cs.ncells,
		omp_get_max_threads());

	float *vel0 = malloc(sizeof(float) * 3 * cs.ncells);
	float *vband = malloc(sizeof(float) * 3 * cs.ncells);
	initial_velocity(&o, &cs, vel0, vband);

	FluidParams fp = { o.forcing, o.rotation, o.bfecc, o.cg_tol, o.cg_maxit, 1 };
	Fluid F;
	fluid_init(&F, &cs, &fp);
	fluid_set_state(&F, vel0, o.forcing > 0 ? vband : NULL);
	free(vel0);
	free(vband);
	fprintf(stderr, "initial state: max speed %.3f, CG %d its (res %.1e)\n",
		fluid_max_speed(&F), F.cg_iters, F.cg_relres);

	Particles P;
	if (particles_init(&P, o.particles))
		return 1;
	particles_color_from_image(&P, img, iw, ih);
	free(img);

	Render R;
	if (render_init(&R, o.res, (float) o.splat_radius, (float) o.trail_alpha))
		return 1;
	fprintf(stderr, "setup: %.2f s\n", omp_get_wtime() - t0);

	double tp = 0, tf = 0, tr = 0, to = 0, simtime = 0;
	long cgsum = 0;
	for (int step = 1; step <= o.steps; step++) {
		double dt = o.dt;
		if (dt <= 0) {
			double vmax = fluid_max_speed(&F);
			dt = o.cfl * cs.h / (vmax > 1e-6 ? vmax : 1e-6);
		}
		double a = omp_get_wtime();
		particles_advect(&P, &cs, F.vel, dt);
		double b = omp_get_wtime();
		fluid_step(&F, dt);
		cgsum += F.cg_iters;
		double c = omp_get_wtime();
		render_fade(&R, o.fade);
		render_deposit(&R, &P);
		double d = omp_get_wtime();
		tp += b - a;
		tf += c - b;
		tr += d - c;
		simtime += dt;
		if (step % o.every == 0 || step == o.steps) {
			write_outputs(&o, &R, step);
			double e = omp_get_wtime();
			to += e - d;
			fprintf(stderr, "step %d/%d  t=%.3f dt=%.4f  CG avg %.1f its  "
				"ms/step: particles %.1f fluid %.1f render %.1f | output %.2f s\n",
				step, o.steps, simtime, dt, (double) cgsum / step,
				1000 * tp / step, 1000 * tf / step, 1000 * tr / step, e - d);
		}
	}
	fprintf(stderr, "total: %.1f s (particles %.1f, fluid %.1f, render %.1f, output %.1f)\n",
		omp_get_wtime() - t0, tp, tf, tr, to);

	render_free(&R);
	particles_free(&P);
	fluid_free(&F);
	cs_free(&cs);
	return 0;
}
