gpu: main
	./matmul

main: matmul.o main.o
	nvcc matmul.o main.o -o matmul -lcublas

matmul.o: matmul.cu matmul.h
	nvcc -O3 -c matmul.cu -o matmul.o -lineinfo -lcublas

main.o: matmul.cpp matmul.h
	g++ -O3 -c matmul.cpp -o main.o
