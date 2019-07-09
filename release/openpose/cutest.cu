


#define N 5

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N)
		C[i][j] = A[i][j] + B[i][j];
}

