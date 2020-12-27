#include "cuda.cu"
#include "CRC_polynomial_cuda_wrapper.h"
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>

using std::string;

vector<uint64_t> read_file(const string& s) {
    std::vector<uint64_t> S;
    std::ifstream f(s);
    std::string line;
	while (std::getline(f, line)) {
	  std::istringstream iss(line);
	  uint64_t v;
	  while (iss >> v) S.push_back(v);
	}
	f.close();
    return move(S);
}

TEST(find_CRC_polynomials, 2_64_8) {
	std::vector<uint64_t> S = read_file("./test_files/output_cuda_t_2_da_64_dc_8");
    std::vector<uint64_t> O = CRC_polynomial_cuda_t2_wrapper<64,8>();
    std::sort(S.begin(), S.end());
    std::sort(O.begin(), O.end());
    ASSERT_TRUE(S == O);
}

TEST(find_CRC_polynomials, 2_64_16) {
	std::vector<uint64_t> S = read_file("./test_files/output_cuda_t_2_da_64_dc_16");
    std::vector<uint64_t> O = CRC_polynomial_cuda_t2_wrapper<64,16>();
    std::sort(S.begin(), S.end());
    std::sort(O.begin(), O.end());
    ASSERT_TRUE(S == O);
}

TEST(find_CRC_polynomials, 2_128_16) {
	std::vector<uint64_t> S = read_file("./test_files/output_cuda_t_2_da_128_dc_16");
    std::vector<uint64_t> O = CRC_polynomial_cuda_t2_wrapper<128,16>();
    std::sort(S.begin(), S.end());
    std::sort(O.begin(), O.end());
    ASSERT_TRUE(S == O);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}