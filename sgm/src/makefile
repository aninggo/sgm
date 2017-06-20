CC=g++
DEST_ARCH=#-m32 -L/usr/lib32
OPENMP= -fopenmp
CFLAGS=-I. -I/usr/include -std=c++11 -msse4.1 -msse4.2 -O3 -ffast-math  -march=core2 $(DEST_ARCH) $(OPENMP) -Wno-write-strings
LDFLAGS= $(DEST_ARCH) $(OPENMP)
ODIR=../obj
DEPS = StereoBMHelper.h
_OBJ = rSGMCmd.o FastFilters.o StereoBMHelper.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

rSGMCmd: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ rSGMCmd
