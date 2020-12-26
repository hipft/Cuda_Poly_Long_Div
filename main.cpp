/*
 * twocoef.cpp
 *
 *  Created on: Oct 22, 2020
 *      Author: alissabrown
 *
 *	Received a lot of help from Anton and the recursive function in the possibleC function is modeled after code found at
 *	https://www.geeksforgeeks.org/print-all-combinations-of-given-length/
 *
 *
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include "bitset.h"
#include <cstdint>

using namespace std;


int remainder_is_nonzero(int da, bool *A, int db, const uint64_t B)
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

int test_all_two_bit_patterns(int da, int dc, const uint64_t C, int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i;
	int j;
	int k;
	int ai, aj;
	int ret;
	bool B[da + dc + 1];
	bool A[da + 1 + dc];

	for (i = 0; i <= da; i++) {
		A[i] = 0;
	}

	for (i = 0; i <= da; i++) {

		for (ai = 1; ai < 2; ai++) {

			A[i] = ai;

			for (j = i + 1; j <= da; j++) {

				for (aj = 1; aj < 2; aj++) {

					A[j] = aj;

					for (k = 0; k <= da; k++) {
						B[dc + k] = A[k];
					}
					for (k = 0; k < dc; k++) {
						B[k] = 0;
					}

					ret = remainder_is_nonzero (da, B, dc, C);

					if (f_v) {
						cout << " : ";
						for (k = dc; k >= 0; k--) {
							cout << B[k];
						}
						cout << endl;
					}

					if (!ret) {
						return false;
					}

				}
				A[j] = 0;
			}

			//cout << endl;
		}
		A[i] = 0;
	}
	return true;
}

int test_all_three_bit_patterns(int da, int dc, const uint64_t C, int verbose_level)
// returns true if division by C leaves a nonzero remainder for all two bit error patters
{

	//cout << "choosetwo" << endl;

	int f_v = (verbose_level >= 1);

	int i1, i2, i3;
	int k;
	int a1, a2, a3;
	int ret;
	bool B[da + dc + 1];
	bool A[da + 1 + dc];

	for (int h = 0; h <= da; h++) {
		A[h] = 0;
	}

	for (i1 = 0; i1 <= da; i1++) {

		for (a1 = 1; a1 < 2; a1++) {

			A[i1] = a1;

			for (i2 = i1 + 1; i2 <= da; i2++) {

				for (a2 = 1; a2 < 2; a2++) {

					A[i2] = a2;

					for (i3 = i2 + 1; i3 <= da; i3++) {

						for (a3 = 1; a3 < 2; a3++) {

							A[i3] = a3;

							for (int h = 0; h <= da; h++) {
								B[dc + h] = A[h];
							}
							for (int h = 0; h < dc; h++) {
								B[h] = 0;
							}

							ret = remainder_is_nonzero (da, B, dc, C);

							if (!ret) {
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

void search_for_CRC_polynomial(int t, int da, int dc, uint64_t& C, int i,
		long int &nb_sol, std::vector<std::vector<int> > &Solutions, int verbose_level)
{

	if (i > dc) {

		int ret;

		if (t >= 2) {
			ret = test_all_two_bit_patterns(da, dc, C, verbose_level);
			if (ret && t >= 3) {
				ret = test_all_three_bit_patterns(da, dc, C, verbose_level);
			}
		}
		else {
			cout << "illegal value for t, t=" << t << endl;
			exit(1);
		}
		if (ret) {
			vector<int> sol;

			for (int j = 0; j <= dc; j++) {
				sol.push_back((C >> (dc-j)) & 1);
			}
			Solutions.push_back(sol);


			nb_sol++;
		}

		return;
	}

	if (i == dc) {
		C |= uint64_t(1);
		search_for_CRC_polynomial(t, da, dc, C, i + 1, nb_sol, Solutions, verbose_level);
		return;
	}

	C &= ~(uint64_t(1)<<(dc-i));
	search_for_CRC_polynomial(t, da, dc, C, i + 1, nb_sol, Solutions, verbose_level);

	C |= uint64_t(1) << (dc-i);
	search_for_CRC_polynomial(t, da, dc, C, i + 1, nb_sol, Solutions, verbose_level);
}

std::vector<std::vector<int>> find_CRC_polynomials(int t, int da, int dc, int verbose_level)
{
	//int dc = 4; //dc is the number of parity bits & degree of g(x)
	//int da = 4; //da is the degree of the information polynomial
	long int nb_sol = 0;

	//int dc2 = dc; //This is also the degree of C/ # of parity bits


	uint64_t C{0}; //Array C (what we divide by)
	//int p = 2; //this is the number of possible coefficients (1 & 0 in this case)

	std::vector<std::vector<int>> Solutions;

	search_for_CRC_polynomial(t, da, dc, C, 0, nb_sol, Solutions, verbose_level - 1);

	cout << "find_CRC_polynomials info=" << da << " check=" << dc << " nb_sol=" << nb_sol << endl;
//	cout << "{";
//	int k=0;
//	for_each(Solutions.begin(), Solutions.end(), [&](const auto& i){
//		cout << "{";
//		for (int j=0; j<i.size(); ++j) {
//			cout << i[j];
//			if (j+1 != i.size()) cout << ",";
//		}
//		cout << "},";
//		if ((++k) % 2 == 0) cout << endl;
//	});
//	cout << "}" << endl;
//	cout << "find_CRC_polynomials info=" << da << " check=" << dc << " nb_sol=" << nb_sol << endl;
	return move(Solutions);
}


int main() {
	int verbose_level = 1;

	std::vector<std::vector<int>> S = find_CRC_polynomials(2, 256, 16, verbose_level);
	return 0;
}
