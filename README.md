# gaseous-gusty-ball

Generates cubemap textures of gas-giant planets. It simulates 2D incompressible
fluid flow on the surface of a sphere and renders the motion of millions of colored
particles as fading, alpha-blended trails into six cubemap face images.

## Build

Requires gcc (with OpenMP) and libpng.

```bash
make            # builds ./gaseous-gusty-ball
make tests      # builds and runs the numerical test suite
```

## Quick start

```bash
python3 tools/make_test_image.py test/bands.png      # synthetic 64x1024 banded strip (needs Pillow)
./gaseous-gusty-ball -i test/bands.png -o out --steps 1000 --every 100 --preview --net
```

The input image is wrapped around the sphere: image **width → longitude**, image
**height → latitude** (top row = north pole). A tall, thin strip of colors works well.
Each particle takes its color from where it starts and keeps it.

## How it works

| Part | Method |
|---|---|
| Grid | Equiangular cubed sphere, 6 × N × N cells (`--grid N`) |
| Fluid | Barotropic vorticity equation: advect vorticity ω, solve ∇²ψ = ω, velocity u = r̂ × ∇ψ (divergence-free by construction) |
| Operators | Exact spherical-Voronoi finite volumes (consistent across the skewed grid and cube edges) |
| Poisson solve | Conjugate gradients, multigrid-preconditioned, warm-started (~3–5 iterations/step) |
| Advection | Semi-Lagrangian along great circles, with BFECC + limiter (sharper vortices) |
| Initial flow | Multi-octave 3D curl noise projected onto the sphere, plus zonal bands `u_east = A·sin(n·colatitude)` |
| Forcing (optional) | Relaxation toward the band vorticity (`--forcing`), Coriolis (`--rotation`) |
| Particles | Fibonacci-lattice start positions, RK2 advection on the sphere |
| Trails | Tent-filtered splats along each step's arc, "over" blending, global exponential fade |

## Output

`--layout cross` (default) writes `gg0.png … gg5.png`, laid out as:

```
   +------+
   |  4   |
   +------+------+------+------+
   |  0   |  1   |  2   |  3   |
   +------+------+------+------+
   |  5   |
   +------+
```

| Face | Center | Image right | Image up |
|---|---|---|---|
| 0 | +Z (lon 0) | +X | +Y (north) |
| 1 | +X | −Z | +Y |
| 2 | −Z | −X | +Y |
| 3 | −X | +Z | +Y |
| 4 | +Y (north pole) | +X | −Z |
| 5 | −Y (south pole) | +X | +Z |

Faces 0→1→2→3 run eastward. The net is seamless when viewed from outside the planet.

`--layout gl` writes `gg-posx.png, gg-negx.png, gg-posy.png, gg-negy.png, gg-posz.png, gg-negz.png`
(OpenGL `GL_TEXTURE_CUBE_MAP` order and orientation, +Y = north). Per face, the image
orientation is identical to the cross layout; only the names and order differ.

- `--sequence` writes numbered frames (`gg-00100-0.png`, …) instead of overwriting.
- `--net` also writes the assembled cross (`gg-net.png`).
- `--preview` also writes an equirectangular preview (`gg-preview.png`).
- `--bg transparent` (default) writes RGBA. `--bg RRGGBB` composites over a solid color and writes RGB.

## Main options

Run `./gaseous-gusty-ball --help` for the full list.

| Option | Default | Meaning |
|---|---|---|
| `--particles M` | 1000000 | particle count |
| `--grid N` | 256 | fluid cells per cube-face edge |
| `--res R` | 1024 | output face size |
| `--bands n` | 8 | number of counter-rotating bands |
| `--band-amp A` | 1.0 | band speed |
| `--noise-amp A` | 0.5 | curl-noise RMS speed |
| `--noise-scale s` | 3.0 | base whorl frequency (smaller = bigger whorls) |
| `--noise-octaves k` | 4 | layers of detail |
| `--noise-lacunarity l` | 2.0 | frequency ratio between octaves |
| `--noise-gain g` | 0.5 | amplitude ratio between octaves |
| `--forcing k` | 0 | relax toward band flow (keeps bands alive) |
| `--rotation W` | 0 | Coriolis rotation rate |
| `--steps S` / `--every K` | 2000 / 50 | run length / output interval |
| `--cfl C` or `--dt T` | 1.0 / auto | time step control |
| `--fade f` | 0.02 | trail opacity lost per step |
| `--trail-alpha a` | 0.35 | opacity of each deposit |
| `--splat-radius r` | 1.0 | trail half-width (texels) |

## Tips

- **Bigger/smaller whorls:** `--noise-scale`. **More fine detail:** more `--noise-octaves`
  or higher `--noise-gain`. The program warns if the finest octave is too small for `--grid`.
- **Keep distinct bands over long runs:** `--forcing 0.2` to `1.0`, optionally `--rotation 5` to `20`.
- **Gaps in coverage:** raise `--particles`, `--splat-radius` or `--trail-alpha`, or lower `--fade`.
- **Speed:** `--grid 128` is about 4× cheaper for the fluid. Particle count and `--res` drive render cost.

## Tests

`make tests` runs:

- `test_cubesphere`: grid round trip, Voronoi neighbor reciprocity, total area = 4π, output-layout edge adjacency.
- `test_noise`: analytic simplex gradient, divergence-free curl noise, whorl size vs `--noise-scale`.
- `test_poisson`: Laplacian convergence on spherical harmonics, PCG solve accuracy, curl(u) = ω.
- `test_flow`: solid-body rotation (particles return after one revolution; ω preserved).

## Origin

gaseous-gusty-ball came to exist from efforts to incorporate fluid dynamics into gaseous-giganticus.  After
several failed attempts to convince various AI agents to add fluid dynamics capabilities into gaseous-giganticus,
I instead opted to simply describe what gaseous-giganticus does to Claude Opus 5.5, with the additional requirement
that it used fluid dynamics to evolve the situation.  After about 30 minutes of work, totally exhausting my
weekly quota for Opus 5.5, plus a bit of Gemini Argon 4 quota, it came up with this.  It did not come up with the
name, I take all the blame for that. -- steve

