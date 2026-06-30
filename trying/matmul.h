void matmul_gpu(int n, int k, int m);
void matmul_gpu_cublas(int n, int k, int m);
void matmul_get(float *OUT, int n, int m);
void matmul_alloc(float *A, float *B, int n, int k, int m);
