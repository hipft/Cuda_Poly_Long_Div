#include <vector>
#include <iostream>
#include <thread>
#include <cassert>
#include <stdint.h>
#include "cuda.cu"

using std::min;
using std::vector;
using std::thread;
using std::cout;
using std::endl;
using std::ref;

#ifndef CRC_polynomial_cuda_wrapper
#define CRC_polynomial_cuda_wrapper

class params {
public:
	vector<uint16_t> solutions;
	uint64_t start, end;
	int block_dim, id;
};

template<int da, int dc>
void CRC_polynomial_cuda_t2_wrapper_thread(params& p, const size_t n) {
    uint64_t grid_dim = (p.end-p.start)/p.block_dim + 1;
    grid_dim = min(grid_dim, 1ul<<13);
    uint64_t total_threads = p.block_dim * grid_dim;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    bool *r, *rh=new bool[total_threads]();
    cudaMalloc(&r, sizeof(bool)*total_threads);

    for (uint64_t k=p.start; k<p.end; k+=total_threads) {
        CRC_polynomial_cuda_t2<da,dc><<<grid_dim,p.block_dim,0,stream>>>(k,p.end,r);
        cudaDeviceSynchronize();
        cudaMemcpy(rh, r, total_threads, cudaMemcpyDeviceToHost);

        for (uint64_t j{0}; j<total_threads; ++j) {
            if (rh[j]) p.solutions.push_back(k+j);
        }
    }

    delete[] rh;
    cudaFree(r);
    cudaStreamDestroy(stream);
}

template<int da, int dc>
vector<uint64_t> CRC_polynomial_cuda_t2_wrapper() {
	assert(dc < 64);
	constexpr int blockDim{1<<6}; // threads per block

	const size_t n = std::thread::hardware_concurrency();
	vector<thread> threads(n);
	vector<params> p(n);

	for (size_t i{0}, d{(1ul<<(dc+1))/n}; i<n; ++i) {
		p[i].id = i;
		p[i].start = i*d;
		p[i].end = (i+1 == n) ? (1ul<<(dc+1)) : (p[i].start + d);
		p[i].block_dim = blockDim;
		threads[i] = thread(CRC_polynomial_cuda_t2_wrapper_thread<da,dc>, ref(p[i]), n);
	}

	for (thread& t : threads) t.join();
	
    vector<uint64_t> Solutions;
    for (params& param : p) 
        Solutions.insert(Solutions.end(), param.solutions.begin(), param.solutions.end());

    return std::move(Solutions);
}

#endif