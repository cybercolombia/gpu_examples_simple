CXX = g++
CXXFLAGS = -std=c++14 -isystem/opt/rocm/include -g -Ofast -march=native -Wall -D__HIP_PLATFORM_AMD__
LDFLAGS = -L/opt/rocm/lib -Wl,-rpath=/opt/rocm/lib
LDLIBS = -lrocblas -lm -lpthread -lstdc++ -lamdhip64

TARGET = ejemplo_rocblas
SRCS = main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)