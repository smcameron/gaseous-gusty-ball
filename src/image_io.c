#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>

#include "image_io.h"

float *png_load_rgba(const char *path, int *w, int *h)
{
	png_image img;
	memset(&img, 0, sizeof(img));
	img.version = PNG_IMAGE_VERSION;
	if (!png_image_begin_read_from_file(&img, path)) {
		fprintf(stderr, "%s: %s\n", path, img.message);
		return NULL;
	}
	img.format = PNG_FORMAT_RGBA;
	size_t n = PNG_IMAGE_SIZE(img);
	unsigned char *buf = malloc(n);
	if (!buf || !png_image_finish_read(&img, NULL, buf, 0, NULL)) {
		fprintf(stderr, "%s: %s\n", path, img.message);
		free(buf);
		png_image_free(&img);
		return NULL;
	}
	*w = (int) img.width;
	*h = (int) img.height;
	size_t np = (size_t) img.width * img.height * 4;
	float *out = malloc(sizeof(float) * np);
	for (size_t i = 0; i < np; i++)
		out[i] = buf[i] / 255.0f;
	free(buf);
	return out;
}

int png_save(const char *path, int w, int h, int channels, const unsigned char *data)
{
	png_image img;
	memset(&img, 0, sizeof(img));
	img.version = PNG_IMAGE_VERSION;
	img.width = (png_uint_32) w;
	img.height = (png_uint_32) h;
	img.format = channels == 4 ? PNG_FORMAT_RGBA : PNG_FORMAT_RGB;
#ifdef PNG_IMAGE_FLAG_FAST
	img.flags |= PNG_IMAGE_FLAG_FAST;
#endif
	if (!png_image_write_to_file(&img, path, 0, data, 0, NULL)) {
		fprintf(stderr, "%s: %s\n", path, img.message);
		return -1;
	}
	return 0;
}
