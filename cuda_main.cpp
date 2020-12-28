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

#include "CRC_polynomial_cuda_wrapper.h"
#include <vector>
#include <sstream>
#include <algorithm>
#include <cassert>
#include "FileWriter.h"
#include "cuda.cu"

using std::vector;
using std::min;


int main() {
	constexpr int da{256};
	constexpr int dc{24};

	assert(dc < 63);

	// for t == 2
#if 1
	cout << "Running t=2" << endl;
	vector<uint64_t> Solutions = CRC_polynomial_cuda_t2_wrapper<da, dc>();
	
	cout << "num_sol: " << Solutions.size() << endl;
	FileWriter fw("output_cuda_t2");
	fw.write(Solutions);
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
