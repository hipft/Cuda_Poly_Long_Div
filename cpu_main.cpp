/*
 * cpu_main.cpp
 *
 *  Created on: Dec 23, 2020
 *      Author: sajeeb
 */


#include "FileWriter.h"
#include "cuda.cu"

class params {
public:
	vector<uint64_t> sol;
	uint64_t start, end;
};

template<int t, int da, int dc>
void search_for_CRC_polynomial_cpu(params& p, FileWriter& fw) {
	if (t<2) return;
	p.start += !(p.start&1ul);
	for (size_t i=p.start; i<p.end; ++++i) {
		if (i > (uint64_t(1)<<(dc+1))-1) break;
		bool ret = test_all_two_bit_patterns<da, dc>(i);
		if (ret && t >= 3) ret = test_all_three_bit_patterns<da, dc>(i);
		if (ret) p.sol.push_back(i);
	}
	fw.write(p.sol);
}

int main() {
	int num_sol{0};

	constexpr int t{2};
	constexpr int da{256};
	constexpr int dc{12};

	FileWriter fw("output_cpu");

	size_t n = std::thread::hardware_concurrency();
	thread threads[n];
	params p[n];

	for (size_t i{0}, d{(uint64_t(1)<<(dc+1))/n}; i<n; ++i) {
		p[i].start = i*d;
		p[i].end = (i+1 == n) ? uint64_t(1)<<(dc+1) : p[i].start + d;
		threads[i] = thread(search_for_CRC_polynomial_cpu<t,da,dc>,
				std::ref(p[i]), std::ref(fw));
	}

	for (size_t i{0}; i<n; ++i) threads[i].join();

	for (size_t i{0}; i<n; ++i) num_sol += p[i].sol.size();
	cout << "num_sol: " << num_sol << endl;
}
