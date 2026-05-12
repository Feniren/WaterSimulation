#include <iostream>

#include "SimulationDebug.cuh"

void DebugSortParticleByHash(int TotalParticleCount, const std::vector<int>& ParticleHashList, const std::vector<int>& ParticleIndexList){
	for (int i = 0; i < TotalParticleCount; i++){
		if (i < 2){
			std::cout << "Sorted Particle " << i << " Hash = " << ParticleHashList[i] << " Original Index = " << ParticleIndexList[i] << std::endl;
		}

		if ((i > 0) && (ParticleHashList[i] < ParticleHashList[i - 1])){
			std::cout << "Sorting failed at " << i << std::endl;
		}
	}
}

void DebugFindCellBoundaries(int TotalParticleCount, int TotalCellCount, const std::vector<int>& ParticleCellStartList, const std::vector<int>& ParticleCellEndList){
	int printed = 0;
	int countedParticles = 0;

	bool valid = true;

	for (int cell = 0; cell < TotalCellCount; cell++){
		int start = ParticleCellStartList[cell];
		int end = ParticleCellEndList[cell];

		if (start == -1 && end == -1){
			continue;
		}

		if (start != -1){
			countedParticles += (end - start);

			if (printed < 2){
				std::cout << "Cell " << cell
					<< " Start = " << start
					<< " End = " << end
					<< " Count = " << (end - start)
					<< std::endl;

				printed++;
			}
		}

		if (start < 0 || end < 0 || start > end || end > TotalParticleCount) {
			valid = false;

			std::cout << "Invalid bounds for cell " << cell
				<< " Start = " << start
				<< " End = " << end
				<< std::endl;

			break;
		}
	}

	if (valid){
		std::cout << "Cell start/end bounds look valid." << std::endl;
	}

	std::cout << "Counted particles from cells = "
		<< countedParticles
		<< " / "
		<< TotalParticleCount
		<< std::endl;
}

void DebugReorderParticles(int TotalParticleCount, const std::vector<float3>& SortedParticlePositionList, const std::vector<int>& ParticleIndexList){
	for (int i = 0; i < 2; i++) {
		int OriginalIndex = ParticleIndexList[i];
		float3 p = SortedParticlePositionList[i];

		std::cout << "Sorted slot " << i
			<< " Original Index = " << OriginalIndex
			<< " Pos = ("
			<< p.x << ", "
			<< p.y << ", "
			<< p.z << ")"
			<< std::endl;
	}
}

void DebugComputeDensity(int TotalParticleCount, const std::vector<float>& ParticleDensityList){
	float minDensity = ParticleDensityList[0];
	float maxDensity = ParticleDensityList[0];
	double sumDensity = 0.0;

	for (int i = 0; i < TotalParticleCount; i++) {
		float rho = ParticleDensityList[i];

		minDensity = std::min(minDensity, rho);
		maxDensity = std::max(maxDensity, rho);
		sumDensity += rho;
	}

	double avgDensity = sumDensity / TotalParticleCount;

	std::cout << "Density min = " << minDensity << std::endl;
	std::cout << "Density max = " << maxDensity << std::endl;
	std::cout << "Density avg = " << avgDensity << std::endl;

	for (int i = 0; i < std::min(2, TotalParticleCount); i++) {
		std::cout << "Particle " << i
			<< " Density = "
			<< ParticleDensityList[i]
			<< std::endl;
	}
}

void DebugComputePressure(int TotalParticleCount, const std::vector<float>& ParticlePressureList, const std::vector<float>& ParticleDensityList){
	float minPressure = ParticlePressureList[0];
	float maxPressure = ParticlePressureList[0];
	double sumPressure = 0.0;
	int positiveCount = 0;

	for (int i = 0; i < TotalParticleCount; i++){
		float p = ParticlePressureList[i];

		minPressure = std::min(minPressure, p);
		maxPressure = std::max(maxPressure, p);
		sumPressure += p;

		if (p > 0.0f){
			positiveCount++;
		}
	}

	double avgPressure = sumPressure / TotalParticleCount;

	std::cout << "Pressure min = " << minPressure << std::endl;
	std::cout << "Pressure max = " << maxPressure << std::endl;
	std::cout << "Pressure avg = " << avgPressure << std::endl;
	std::cout << "Positive pressure particles = "
		<< positiveCount
		<< " / "
		<< TotalParticleCount
		<< std::endl;

	for (int i = 0; i < std::min(2, TotalParticleCount); i++){
		std::cout << "Particle " << i
			<< " Density = " << ParticleDensityList[i]
			<< " Pressure = " << ParticlePressureList[i]
			<< std::endl;
	}
}

void DebugComputeForces(int TotalParticleCount, const std::vector<float3>& ParticleForceList){
	float minForceMag = FLT_MAX;
	float maxForceMag = 0.0f;
	double sumForceMag = 0.0;

	for (int i = 0; i < TotalParticleCount; i++){
		float3 f = ParticleForceList[i];

		float mag = sqrtf(
			f.x * f.x +
			f.y * f.y +
			f.z * f.z
		);

		minForceMag = std::min(minForceMag, mag);
		maxForceMag = std::max(maxForceMag, mag);
		sumForceMag += mag;
	}

	double avgForceMag = sumForceMag / TotalParticleCount;

	std::cout << "Force magnitude min = " << minForceMag << std::endl;
	std::cout << "Force magnitude max = " << maxForceMag << std::endl;
	std::cout << "Force magnitude avg = " << avgForceMag << std::endl;

	for (int i = 0; i < std::min(2, TotalParticleCount); i++){
		float3 f = ParticleForceList[i];

		std::cout << "Particle " << i
			<< " Force = ("
			<< f.x << ", "
			<< f.y << ", "
			<< f.z << ")"
			<< std::endl;
	}
}

void DebugIntegrate(int TotalParticleCount, const std::vector<float3>& ParticlePositionList, const std::vector<float3>& ParticleVelocityList){
	float3 minPos = ParticlePositionList[0];
	float3 maxPos = ParticlePositionList[0];

	float maxVelMag = 0.0f;
	double avgVelMag = 0.0;

	for (int i = 0; i < TotalParticleCount; i++) {
		float3 p = ParticlePositionList[i];
		float3 v = ParticleVelocityList[i];

		minPos.x = std::min(minPos.x, p.x);
		minPos.y = std::min(minPos.y, p.y);
		minPos.z = std::min(minPos.z, p.z);

		maxPos.x = std::max(maxPos.x, p.x);
		maxPos.y = std::max(maxPos.y, p.y);
		maxPos.z = std::max(maxPos.z, p.z);

		float velMag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

		maxVelMag = std::max(maxVelMag, velMag);
		avgVelMag += velMag;
	}

	avgVelMag /= TotalParticleCount;

	std::cout << "Position min = ("
		<< minPos.x << ", "
		<< minPos.y << ", "
		<< minPos.z << ")"
		<< std::endl;

	std::cout << "Position max = ("
		<< maxPos.x << ", "
		<< maxPos.y << ", "
		<< maxPos.z << ")"
		<< std::endl;

	std::cout << "Velocity max magnitude = "
		<< maxVelMag
		<< std::endl;

	std::cout << "Velocity avg magnitude = "
		<< avgVelMag
		<< std::endl;
}