CC = gcc
CFLAGS ?= -O3 -march=native -fopenmp -Wall -Wextra -std=gnu11
LDLIBS = -lpng -lm

LIBSRC = src/cubesphere.c src/noise.c src/fluid.c src/particles.c src/render.c src/image_io.c
LIBOBJ = $(LIBSRC:.c=.o)
HDRS = $(wildcard src/*.h)
TESTS = tests/test_cubesphere tests/test_noise tests/test_poisson tests/test_flow

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
MANDIR ?= $(PREFIX)/share/man/man1

.PHONY: all tests install clean

all: gaseous-gusty-ball

gaseous-gusty-ball: src/main.o $(LIBOBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

src/%.o: src/%.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

tests/test_%: tests/test_%.c $(LIBOBJ) $(HDRS)
	$(CC) $(CFLAGS) -Isrc -o $@ $< $(LIBOBJ) $(LDLIBS)

tests: $(TESTS)
	@for t in $(TESTS); do echo "=== $$t"; ./$$t || exit 1; done
	@echo "All tests passed."

clean:
	rm -f gaseous-gusty-ball src/*.o $(TESTS)

install: gaseous-gusty-ball
	install -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(MANDIR)
	install -m 755 gaseous-gusty-ball $(DESTDIR)$(BINDIR)/gaseous-gusty-ball
	install -m 644 gaseous-gusty-ball.1 $(DESTDIR)$(MANDIR)/gaseous-gusty-ball.1

