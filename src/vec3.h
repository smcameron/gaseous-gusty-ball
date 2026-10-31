/* Small inline 3D vector library (double precision). */
#ifndef VEC3_H
#define VEC3_H

#include <math.h>

typedef struct { double x, y, z; } vec3;

static inline vec3 v3(double x, double y, double z) { vec3 r = { x, y, z }; return r; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 v3_scale(vec3 a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline double v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b)
{
	return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline double v3_len(vec3 a) { return sqrt(v3_dot(a, a)); }
static inline vec3 v3_normalize(vec3 a)
{
	double l = v3_len(a);
	return l > 0.0 ? v3_scale(a, 1.0 / l) : a;
}
/* Remove the component of v along unit normal n (project onto tangent plane). */
static inline vec3 v3_tangent(vec3 v, vec3 n) { return v3_sub(v, v3_scale(n, v3_dot(v, n))); }
/* Angle between two vectors (robust for small angles). */
static inline double v3_angle(vec3 a, vec3 b) { return atan2(v3_len(v3_cross(a, b)), v3_dot(a, b)); }
static inline vec3 v3_load(const float *f) { return v3(f[0], f[1], f[2]); }
static inline void v3_store(float *f, vec3 a) { f[0] = (float) a.x; f[1] = (float) a.y; f[2] = (float) a.z; }

/* Move a point p on the unit sphere along the great circle in the direction of
 * tangent velocity v for time s (s may be negative). */
static inline vec3 sphere_move(vec3 p, vec3 v, double s)
{
	vec3 t = v3_tangent(v, p);
	double sp = v3_len(t);
	if (sp < 1e-300)
		return p;
	double th = sp * s;
	vec3 r = v3_add(v3_scale(p, cos(th)), v3_scale(t, sin(th) / sp));
	return v3_normalize(r);
}

/* Longitude/latitude convention: +Y is north, longitude 0 is at +Z, and
 * longitude increases toward +X (east). */
static inline void dir_to_lonlat(vec3 d, double *lon, double *lat)
{
	*lon = atan2(d.x, d.z);
	double y = d.y;
	if (y > 1.0) y = 1.0;
	if (y < -1.0) y = -1.0;
	*lat = asin(y);
}

static inline vec3 lonlat_to_dir(double lon, double lat)
{
	return v3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
}

#endif
