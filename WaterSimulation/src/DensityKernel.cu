#include "DensityKernel.cuh"

#include "HashKernel.cuh"

__host__ __device__ float Poly6Kernel(float DistanceSquared, float SmoothingRadiusSquared, float Poly6Coefficient){
	if (DistanceSquared >= SmoothingRadiusSquared){
		return 0.0f;
	}

	float x = SmoothingRadiusSquared - DistanceSquared;

	return Poly6Coefficient * x * x * x;
}

__global__ void ComputeDensity(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	float* ParticleDensityList,
	const int* ParticleCellStartList,
	const int* ParticleCellEndList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	float ParticleMass,
	float Poly6Coefficient,
	int3 CellGridResolution
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	float3 PositionI = ParticlePositionList[ThreadID];

	int3 BaseCell = CalculateCellGridPosition(PositionI, BoxMin, CellSize);

	float SmoothingRadiusSquared = SmoothingRadius * SmoothingRadius;

	float Density = 0.0f;

	for (int dz = -1; dz <= 1; dz++){
		for (int dy = -1; dy <= 1; dy++){
			for (int dx = -1; dx <= 1; dx++){
				int3 NeighborCell = make_int3(
					BaseCell.x + dx,
					BaseCell.y + dy,
					BaseCell.z + dz
				);

				int NeighborHash = CalculateCellGridHash(NeighborCell, CellGridResolution, false);

				if (NeighborHash == -1){
					continue;
				}

				int Start = ParticleCellStartList[NeighborHash];

				if (Start == -1){
					continue;
				}

				int End = ParticleCellEndList[NeighborHash];

				for (int SortedJ = Start; SortedJ < End; SortedJ++){
					float3 PositionJ = ParticlePositionList[SortedJ];

					float3 r = make_float3(
						PositionI.x - PositionJ.x,
						PositionI.y - PositionJ.y,
						PositionI.z - PositionJ.z
					);

					float DistanceSquared = r.x * r.x + r.y * r.y + r.z * r.z;
					
					Density += ParticleMass * Poly6Kernel(DistanceSquared, SmoothingRadiusSquared, Poly6Coefficient);
				}
			}
		}
	}

	ParticleDensityList[ThreadID] = Density;
}

void ComputeDensity(
	int TotalParticleCount,
	const std::vector<float3>& ParticlePositionList,
	std::vector<float>& ParticleDensityList,
	const std::vector<int>& ParticleCellStartList,
	const std::vector<int>& ParticleCellEndList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	float ParticleMass,
	float Poly6Coefficient,
	int3 CellGridResolution
){
	float SmoothingRadiusSquared = SmoothingRadius * SmoothingRadius;

	for (int i = 0; i < TotalParticleCount; i++){
		float3 PositionI = ParticlePositionList[i];

		int3 BaseCell = CalculateCellGridPosition(PositionI, BoxMin, CellSize);

		float Density = 0.0f;

		for (int dz = -1; dz <= 1; dz++){
			for (int dy = -1; dy <= 1; dy++){
				for (int dx = -1; dx <= 1; dx++){
					int3 NeighborCell = make_int3(
						BaseCell.x + dx,
						BaseCell.y + dy,
						BaseCell.z + dz
					);

					int NeighborHash = CalculateCellGridHash(NeighborCell, CellGridResolution, false);

					if (NeighborHash == -1){
						continue;
					}

					int Start = ParticleCellStartList[NeighborHash];

					if (Start == -1){
						continue;
					}

					int End = ParticleCellEndList[NeighborHash];

					for (int SortedJ = Start; SortedJ < End; SortedJ++){
						float3 PositionJ = ParticlePositionList[SortedJ];

						float3 r = make_float3(
							PositionI.x - PositionJ.x,
							PositionI.y - PositionJ.y,
							PositionI.z - PositionJ.z
						);

						float DistanceSquared = r.x * r.x + r.y * r.y + r.z * r.z;

						Density += ParticleMass * Poly6Kernel(DistanceSquared, SmoothingRadiusSquared, Poly6Coefficient);
					}
				}
			}
		}

		ParticleDensityList[i] = Density;
	}
}