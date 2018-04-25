#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <algorithm>
#include <cstdlib>
#include <curand.h>
#include <chrono>

static const int WORK_SIZE = 1000000;

int main()
{
	std::chrono::steady_clock::time_point gpuStart, gpuStop, cpuStart, cpuStop;
	// TODO: Define a vector of doubles of size WORK_SIZE on the device.
	thrust::device_vector<double> data(WORK_SIZE);

	// Generate random number in parallel on the GPU using curand
	curandGenerator_t gen;
	// We need a "raw" pointer to the space allocated above to pass it to curand
	double* data_raw = thrust::raw_pointer_cast(data.data());
	// Initialize the random number generator and generate the sequence.
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateUniformDouble(gen, data_raw, data.size());

	// TODO: Copy random numbers to CPU
	thrust::host_vector<double> data_host = data;

	gpuStart = std::chrono::steady_clock::now();
	// TODO: Sort random numbers in parallel on GPU
	thrust::sort(data.begin(), data.end());
	gpuStop = std::chrono::steady_clock::now();

	// Copy sorted sequence to CPU
	thrust::host_vector<double> data_sorted = data;

	cpuStart = std::chrono::steady_clock::now();
	// Sort random number on CPU
	thrust::sort(data_host.begin(), data_host.end());
	cpuStop = std::chrono::steady_clock::now();

	// Compare sorted sequences
	for (int i = 0; i < data_host.size(); ++i){
		if (data_sorted[i] != data_host[i])
			std::cout << "Element " << i << ": " << data_sorted[i] << "!=" << data_host[i] << "\n";
	}
	std::cout << data_host[0] << " is the smallest and " << data_host[WORK_SIZE -1] << " is the largest element.\n";

	std::cout << "Time spent sorting" << std::endl;
	std::cout << "  GPU: " << std::chrono::duration<double>(gpuStop - gpuStart).count() << " s" << std::endl;
	std::cout << "  CPU: " << std::chrono::duration<double>(cpuStop - cpuStart).count() << " s" << std::endl;

	// Clean up
	curandDestroyGenerator(gen);
	return 0;
}
