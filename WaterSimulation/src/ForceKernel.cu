#include "ForceKernel.cuh"

#include "HashKernel.cuh"

__device__ float3 AddFloat3(float3 a, float3 b){
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 SubtractFloat3(float3 a, float3 b){
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 MultiplyFloat3(float3 a, float b){
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float DotProductFloat3(float3 a, float3 b){
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float LengthFloat3(float3 a){
	return sqrtf(DotProductFloat3(a, a));
}

__device__ float SpikyGradientMagnitude(
	float r,
	float h,
	float SpikyGradientCoefficient
){
	if (r <= 0.0f || r >= h){
		return 0.0f;
	}

	float x = h - r;

	return SpikyGradientCoefficient * x * x;
}

__device__ float ViscosityLaplacian(
	float r,
	float h,
	float ViscosityLaplacianCoefficient
){
	if (r >= h){
		return 0.0f;
	}

	return ViscosityLaplacianCoefficient * (h - r);
}

__global__ void ComputeForces(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	const float3* ParticleVelocityList,
	const float* ParticleDensityList,
	const float* ParticlePressureList,
	float3* ParticleForceList,
	const int* ParticleCellStartList,
	const int* ParticleCellEndList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	float ParticleMass,
	float Viscosity,
	float3 GravityForce,
	float SpikyGradientCoefficient,
	float ViscosityLaplacianCoefficient,
	int3 GridResolution
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	float3 PositionI = ParticlePositionList[ThreadID];
	float3 VelocityI = ParticleVelocityList[ThreadID];

	float DensityI = fmaxf(ParticleDensityList[ThreadID], 1e-6f);
	float PressureI = ParticlePressureList[ThreadID];

	int3 BaseCell = CalculateCellGridPosition(PositionI, BoxMin, CellSize);

	float SmoothingRadiusSquared = SmoothingRadius * SmoothingRadius;

	float3 PressureForce = make_float3(0.0f, 0.0f, 0.0f);
	float3 ViscosityForce = make_float3(0.0f, 0.0f, 0.0f);

	for (int dz = -1; dz <= 1; dz++){
		for (int dy = -1; dy <= 1; dy++){
			for (int dx = -1; dx <= 1; dx++){
				int3 NeighborCell = make_int3(
					BaseCell.x + dx,
					BaseCell.y + dy,
					BaseCell.z + dz
				);

				int NeighborHash = CalculateCellGridHash(NeighborCell, GridResolution, false);

				if (NeighborHash == -1){
					continue;
				}

				int Start = ParticleCellStartList[NeighborHash];

				if (Start == -1){
					continue;
				}

				int End = ParticleCellEndList[NeighborHash];

				for (int i = Start; i < End; i++){
					if (i == ThreadID){
						continue;
					}

					float3 PositionJ = ParticlePositionList[i];

					float3 RVector = make_float3(
						PositionI.x - PositionJ.x,
						PositionI.y - PositionJ.y,
						PositionI.z - PositionJ.z
					);

					float DistanceSquared =
						RVector.x * RVector.x +
						RVector.y * RVector.y +
						RVector.z * RVector.z;

					if (DistanceSquared >= SmoothingRadiusSquared){
						continue;
					}

					float r = sqrtf(DistanceSquared);

					if (r <= 1e-6f){
						continue;
					}

					float DensityJ = fmaxf(ParticleDensityList[i], 1e-6f);
					float PressureJ = ParticlePressureList[i];

					float3 Direction = make_float3(
						RVector.x / r,
						RVector.y / r,
						RVector.z / r
					);

					float GradientMagnitude = SpikyGradientMagnitude(
						r,
						SmoothingRadius,
						SpikyGradientCoefficient
					);

					float PressureScalar =
						-ParticleMass *
						(PressureI + PressureJ) *
						0.5f *
						(1.0f / DensityJ) *
						GradientMagnitude;
					
					PressureForce.x += PressureScalar * Direction.x;
					PressureForce.y += PressureScalar * Direction.y;
					PressureForce.z += PressureScalar * Direction.z;

					float Laplacian = ViscosityLaplacian(
						r,
						SmoothingRadius,
						ViscosityLaplacianCoefficient
					);

					float3 VelocityJ = ParticleVelocityList[i];

					float3 VelocityDifference = make_float3(
						VelocityJ.x - VelocityI.x,
						VelocityJ.y - VelocityI.y,
						VelocityJ.z - VelocityI.z
					);

					float ViscosityScalar =
						Viscosity *
						ParticleMass *
						(1.0f / DensityJ) *
						Laplacian;

					ViscosityForce.x += ViscosityScalar * VelocityDifference.x;
					ViscosityForce.y += ViscosityScalar * VelocityDifference.y;
					ViscosityForce.z += ViscosityScalar * VelocityDifference.z;
				}
			}
		}
	}

	float3 Gravity = make_float3(
		DensityI * GravityForce.x,
		DensityI * GravityForce.y,
		DensityI * GravityForce.z
	);

	ParticleForceList[ThreadID] = make_float3(
		PressureForce.x + ViscosityForce.x + Gravity.x,
		PressureForce.y + ViscosityForce.y + Gravity.y,
		PressureForce.z + ViscosityForce.z + Gravity.z
	);
}