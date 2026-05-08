#include "NeighborSearchKernel.cuh"

#include "HashKernel.cuh"

__global__ void CountNeighbors(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	const int* ParticleCellStartList,
	const int* ParticleCellEndList,
	int* ParticleNeighborCountList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	int3 CellGridResolution
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	float3 PositionI = ParticlePositionList[ThreadID];
	int3 BaseCell = CalculateCellGridPosition(PositionI, BoxMin, CellSize);

	float SmoothingRadiusSquared = SmoothingRadius * SmoothingRadius;
	int Count = 0;

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

				for (int SortedJ = Start; SortedJ < End; SortedJ++)
				{
					float3 PositionJ = ParticlePositionList[SortedJ];

					float3 r = make_float3(
						PositionI.x - PositionJ.x,
						PositionI.y - PositionJ.y,
						PositionI.z - PositionJ.z
					);

					float DistanceSquared = r.x * r.x + r.y * r.y + r.z * r.z;

					if (DistanceSquared < SmoothingRadiusSquared){
						Count++;
					}
				}
			}
		}
	}
	
	ParticleNeighborCountList[ThreadID] = Count;
}