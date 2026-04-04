VERILATOR_ROOT = /usr/local/share/verilator

CXX = g++
CXXFLAGS = -std=c++20 -O3 -g

# Verilator headers
VINC = -Iobj_dir -I$(VERILATOR_ROOT)/include

# SDL2 flags (preferred)
SDL2_CFLAGS = $(shell sdl2-config --cflags)
SDL2_LIBS   = $(shell sdl2-config --libs)

# Library paths (if you really need /usr/local/lib)
LIBPATHS = -L/usr/local/lib

all: run

run: main.cpp obj_dir/verilated.o obj_dir/verilated_vcd_c.o obj_dir/VIntegrate__ALL.a
	$(CXX) $(CXXFLAGS) $(VINC) $(SDL2_CFLAGS) -o $@ $^ $(LIBPATHS) $(SDL2_LIBS)

obj_dir/%.o: $(VERILATOR_ROOT)/include/%.cpp
	$(CXX) $(CXXFLAGS) $(VINC) $(SDL2_CFLAGS) -c $< -o $@

clean:
	rm -f obj_dir/* run

