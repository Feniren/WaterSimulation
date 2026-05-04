#include "DensityKernel.cuh"

__global__ void ComputeDensity(int n, float3* Position, float* Density, float h, float Mass){
	int Index = blockIdx.x * blockDim.x + threadIdx.x;

	if (Index >= n){
		return;
	}

	float3 Pi = Position[Index];
	float Rho = 0.0f;

	for (int i = 0; i < n; i++){
		float3 rij = make_float3(
			Pi.x - Position[i].x,
			Pi.y - Position[i].y,
			Pi.z - Position[i].z);

		float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
		Rho += Mass * poly6Kernel(r2, h);
	}

	Density[Index] = Rho;
}