/**
 * Copyright 2020 Sajeeb Roy Chowdhury
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software
 * is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <sstream>
#include <algorithm>
#include <cassert>
#include "FileWriter.h"
#include "cuda.cu"

using std::min;

class params {
public:
	uint64_t start, end;
	int block_dim, id, num_sol{0};
};

template<int da, int dc>
void CRC_polynomial_cuda_t2_wrapper(params& p, FileWriter& fw, const int& n) {
	uint64_t grid_dim = (p.end-p.start)/p.block_dim + 1;
	grid_dim = min(grid_dim, ((uint64_t(1)<<31)-1)/std::thread::hardware_concurrency());
	uint64_t total_threads = p.block_dim * grid_dim;
	if (p.id==n-1) cout << "blocks: " << p.block_dim << ", grids: " << grid_dim << endl;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	bool *r, *rh=new bool[total_threads]();
	cudaMalloc(&r, sizeof(bool)*total_threads);

	for (uint64_t i=p.start; i<p.end; i+=total_threads) {
		CRC_polynomial_cuda_t2<da,dc><<<grid_dim,p.block_dim,0,stream>>>(i,p.end,r);
		cudaDeviceSynchronize();
		cudaMemcpy(rh, r, total_threads, cudaMemcpyDeviceToHost);

		vector<uint64_t> S;
		for (uint64_t j{0}; j<total_threads; ++j) {
			if (rh[j]) S.push_back(i+j);
		}
		p.num_sol += S.size();
		fw.write(S);
	}

	delete[] rh;
	cudaFree(r);
	cudaStreamDestroy(stream);
}

int main() {
	uint64_t ns{0};
	constexpr int da{128};
	constexpr int dc{16};

	assert(dc < 64);

	// for t == 2
#if 1
	cout << "Running t=2" << endl;
	constexpr int blockDim{1<<6}; // threads per block

	FileWriter fw("output_cuda_t2");
	size_t n = std::thread::hardware_concurrency();
	thread threads[n];
	params p[n];

	for (size_t i{0}, d{(uint64_t(1)<<(dc+1))/n}; i<n; ++i) {
		p[i].id = i;
		p[i].start = i*d;
		p[i].end = (i+1 == n) ? uint64_t(1)<<(dc+1) : p[i].start + d;
		p[i].block_dim = blockDim;
		threads[i] = thread(CRC_polynomial_cuda_t2_wrapper<da,dc>,
				std::ref(p[i]), std::ref(fw), std::ref(n));
	}

	for (size_t i{0}; i<n; ++i) threads[i].join();
	for (size_t i{0}; i<n; ++i) ns += p[i].num_sol;
	cout << "num_sol: " << ns << endl;
#else
	// for t == 3
	cout << "Running t=3" << endl;
	std::ifstream f ("output_cuda_t2");
	vector<uint64_t> S;
	string line;
	while (std::getline(f, line)) {
	  std::istringstream iss(line);
	  uint64_t v;
	  while (iss >> v) S.push_back(v);
	}
	f.close();
	cout << "number of t2 solutions: " << S.size() << endl;

	// copy to GPU
	uint64_t* data;
	cudaMalloc(&data, sizeof(uint64_t)*S.size());
	cudaMemcpy(data, S.data(), sizeof(uint64_t)*S.size(), cudaMemcpyHostToDevice);

	// allocate for result
	bool* d_res;
	cudaMalloc(&d_res, sizeof(bool)*S.size());

	// compute
	CRC_polynomial_cuda_t3<da,dc><<<S.size()/32+1, 32>>>(data, d_res, S.size());
	bool* h_res = new bool[sizeof(bool)*S.size()]();

	// copy result
	cudaDeviceSynchronize();
	cudaMemcpy(h_res, d_res, sizeof(bool)*S.size(), cudaMemcpyDeviceToHost);

	vector<uint64_t> final_result;
	for (uint64_t i{0}; i<S.size(); ++i)
		if (h_res[i]) final_result.push_back(S[i]);
	ns = final_result.size();

	FileWriter fw("output_cuda_t3");
	fw.write(final_result);

	cudaFree(data);
	cudaFree(d_res);
	delete[] h_res;

	cout << "nb_sol: " << ns << endl;
#endif
}
