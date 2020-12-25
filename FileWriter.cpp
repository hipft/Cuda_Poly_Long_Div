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

#include "FileWriter.h"

FileWriter::FileWriter(const string& filename) {
	_file_.open(filename, std::ofstream::out | std::ofstream::app);
	kill.store(false);
	terminated.store(false);
	t = thread(&FileWriter::write_data, this);
	t.detach();
}

FileWriter::~FileWriter() {
	kill.store(true);
	while (!terminated.load()) cv.notify_all(); // bc. notify signals can get lost
	while (!data.empty()) {
		for_each(data.front().begin(), data.front().end(), [&](const int& i){
			_file_ << i << " ";
		});
		data.pop_front();
	}
	_file_.close();
}

void FileWriter::write_data() {
	std::unique_lock<mutex> lock(m);
	vector<uint64_t> v;
	while (!kill.load()) {
		cv.wait(lock, [&]{
			if (!data.empty()) {
				v = move(data.front());
				data.pop_front();
				return true; // release lock and exit cv pred. block
			}
			return kill.load();
		});
		for_each(v.begin(), v.end(), [&](const int& i){
			_file_ << i << " ";
		});
		_file_.flush();
	}
	terminated.store(true);
}

void FileWriter::write(const vector<uint64_t>& v) {
	m.lock();
	data.push_back(move(v));
	m.unlock();
	cv.notify_all();
}
