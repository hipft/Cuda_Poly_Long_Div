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

#include <cuda.h>

__device__ __host__
int remainder_is_nonzero(const int& da, bool* A, const int& db, const uint64_t& B)
// returns true if the remainder of A after division by B is nonzero
{
	for (int i = da + db; i >= db; i--) {
		if (A[i]) {
			for (int j = db, k = i; j >= 0; j--, k--) {
				A[k] = (A[k] + ((B >> (db-j))&1)) & 1;
			}
		}
	}
	for (int k = da + db; k >= 0; k--) {
		if (A[k]) {
			return true;
		}
	}
	return false;
}

template<int da, int dc>
__device__ __host__
bool test_all_two_bit_patterns(const uint64_t& C)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{
	bool B[da + dc + 1];
	bool A[da + 1 + dc];

	memset(A, 0, da + dc + 1);
	memset(B, 0, da + dc + 1);

	for (int i = 0; i <= da; i++) {
		for (int ai = 1; ai < 2; ai++) {
			A[i] = ai;
			for (int j = i + 1; j <= da; j++) {
				for (int aj = 1; aj < 2; aj++) {
					A[j] = aj;
					for (int k = 0; k <= da; k++) {
						B[dc + k] = A[k];
					}
					for (int k = 0; k < dc; k++) {
						B[k] = 0;
					}
					if (!remainder_is_nonzero (da, B, dc, C)) {
						return false;
					}
				}
				A[j] = 0;
			}
		}
		A[i] = 0;
	}
	return true;
}

template<int da, int dc>
__device__ __host__
bool test_all_three_bit_patterns(const uint64_t& C)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{
	bool B[da + dc + 1];
	bool A[da + 1 + dc];

	memset(A, 0, da + dc + 1);
	memset(B, 0, da + dc + 1);

	for (int i1 = 0; i1 <= da; i1++) {
		for (int a1 = 1; a1 < 2; a1++) {
			A[i1] = a1;
			for (int i2 = i1 + 1; i2 <= da; i2++) {
				for (int a2 = 1; a2 < 2; a2++) {
					A[i2] = a2;
					for (int i3 = i2 + 1; i3 <= da; i3++) {
						for (int a3 = 1; a3 < 2; a3++) {
							A[i3] = a3;
							for (int h = 0; h <= da; h++) {
								B[dc + h] = A[h];
							}
							for (int h = 0; h < dc; h++) {
								B[h] = 0;
							}
							if (!remainder_is_nonzero (da, B, dc, C)) {
								return false;
							}
						}
						A[i3] = 0;
					}
				}
				A[i2] = 0;
			}
		}
		A[i1] = 0;
	}
	return true;
}

template<int da, int dc>
__global__
void CRC_polynomial_cuda_t2(uint64_t C, uint64_t e, bool* res) {
	uint64_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	bool ret = false;
	res[thread_id] = ret;

	C += thread_id;

	if (!(C&1)) return;
	if (C >= e) return;
	if (C > (uint64_t(1)<<(dc+1))-1) return;

	ret = test_all_two_bit_patterns<da, dc>(C);
	res[thread_id] = ret;
}

template<int da, int dc>
__global__
void CRC_polynomial_cuda_t3(uint64_t* data, bool* res, size_t size) {
	uint64_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_id > size) return;
	res[thread_id] = false;
	res[thread_id] = test_all_three_bit_patterns<da, dc>(data[thread_id]);
}
