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

#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <condition_variable>

using std::vector;
using std::ofstream;
using std::string;
using std::thread;
using std::mutex;
using std::deque;
using std::move;
using std::atomic;
using std::for_each;
using std::condition_variable;
using std::cout;
using std::endl;

#ifndef FILEWRITER_H_
#define FILEWRITER_H_

class FileWriter {
	deque<vector<uint64_t>> data;
	atomic<bool> kill;
	atomic<bool> terminated;
	ofstream _file_;
	mutex m;
	condition_variable cv;
	thread t;

	void write_data();

public:
	FileWriter(const string& filename);
	virtual ~FileWriter();

	void write(const vector<uint64_t>& v);
};

#endif /* FILEWRITER_H_ */
