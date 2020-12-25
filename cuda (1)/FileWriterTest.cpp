/*
 * FileWriterTest.cpp
 *
 *  Created on: Dec 23, 2020
 *      Author: sajeeb
 */


#include "FileWriter.h"

#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

using namespace std;

int main() {
	FileWriter fw("sample_file");
	for (int i=0; i<10; ++i) {
		fw.write({1,2,3,4,5,6,7,8,9});
		std::this_thread::sleep_for (std::chrono::seconds(5));
		cout << i << endl;
	}
}
