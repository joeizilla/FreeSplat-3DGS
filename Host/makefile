CXX := g++
CXXFLAGS := -std=c++20 -O3 -g -Wall -Wextra -Wpedantic

TARGET := run
SRCS := main.cpp
OBJS := $(SRCS:.cpp=.o)

# SDL2 flags
SDL2_CFLAGS := $(shell sdl2-config --cflags 2>/dev/null)
SDL2_LIBS   := $(shell sdl2-config --libs   2>/dev/null)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(SDL2_LIBS) -lHalf

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(SDL2_CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
