NVCC      := nvcc
CXX       := g++
NVCCFLAGS := -O3 -lineinfo
CXXFLAGS  := -O3

matmul: kernels.o driver.o
	$(NVCC) kernels.o driver.o -o matmul -lcublas

kernels.o: src/kernels.cu src/matmul.h
	$(NVCC) $(NVCCFLAGS) -c src/kernels.cu -o kernels.o

driver.o: src/driver.cpp src/matmul.h
	$(CXX) $(CXXFLAGS) -c src/driver.cpp -o driver.o

# build + run the yours-vs-cuBLAS benchmark sweep
run: matmul
	./matmul

# definitive float64 correctness check (needs uv + torch)
oracle: matmul
	./matmul >/dev/null
	uv run --with numpy --with torch python bench/oracle.py

clean:
	rm -f matmul kernels.o driver.o A.bin B.bin C.bin

.PHONY: run oracle clean
