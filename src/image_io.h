/* PNG input/output via libpng's simplified API. */
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

/* Returns malloc'd float RGBA [0,1] pixels (row 0 = top), or NULL. */
float *png_load_rgba(const char *path, int *w, int *h);
/* channels = 3 (RGB) or 4 (RGBA); data is 8-bit, row 0 = top. Returns 0 on success. */
int png_save(const char *path, int w, int h, int channels, const unsigned char *data);

#endif
