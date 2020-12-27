CC=g++
CC_FLAGS=-std=c++14 -Wall -O3
CC_LIB=-lpthread
CUDA_LIB=-lcuda -lcudart -lcudadevrt
CUDA_FLAGS= -gencode arch=compute_60,code=sm_60 \
			-gencode arch=compute_61,code=sm_61 \
			-gencode arch=compute_75,code=sm_75 \
			-Xptxas -O3 \
			-Xcompiler -O3 \
			-Xcompiler -march=native \
			-Xcompiler -mtune=native \
			-Xcompiler -funroll-loops \
			-Xcompiler -fgcse-lm \
			-Xcompiler -ftree-vectorize \
			-Xcompiler -mavx \
			-Xcompiler -mfpmath=both

all: cuda cpu_main

cuda: cuda_main.cpp \
			cuda.cu \
			FileWriter.cpp \
			FileWriter.h \
			CRC_polynomial_cuda_wrapper.h \
			FileWriterTest.cpp \
			makefile
	$(CC) $(CC_FLAGS) -c ./FileWriter.cpp -o FileWriter.o $(CC_LIB)
	# this is because cuda_main contains both cuda and c++ libs
	nvcc -x cu -c $(CUDA_FLAGS) -rdc=true cuda_main.cpp -o cuda_main.o
	nvcc $(CUDA_FLAGS) FileWriter.o cuda_main.o -o cuda $(CUDA_LIB) $(CC_LIB)

cpu_main: cpu_main.cpp cuda.cu FileWriter.h FileWriter.cpp
	$(CC) $(CC_FLAGS) -c FileWriter.cpp -o FileWriter.o $(CC_LIB)
	nvcc -x cu -c $(CUDA_FLAGS) -rdc=true cpu_main.cpp -o cpu_main.o
	nvcc $(CUDA_FLAGS) FileWriter.o cpu_main.o -o cpu $(CUDA_LIB) $(CC_LIB)
	
cuda_test: cuda cuda_test.cpp
	nvcc -x cu -c $(CUDA_FLAGS) -rdc=true cuda_test.cpp -o cuda_test.o
	nvcc $(CUDA_FLAGS) cuda_test.o -o cuda_test $(CUDA_LIB) $(CC_LIB) -lgtest -L/usr/lib

FileWriter: FileWriter.cpp FileWriter.h FileWriterTest.cpp
	$(CC) -std=c++14 -Wall -O3 -c ./FileWriter.cpp -o FileWriter.o
	$(CC) -std=c++14 -Wall -O3 -c ./FileWriterTest.cpp -o FileWriterTest.o
	$(CC) -std=c++14 -Wall -O3 ./FileWriter.o ./FileWriterTest.o -o FileWriterTest -lpthread
	
test: test.cpp main.cpp
	$(CC) -c test.cpp -o test.o
	$(CC) -o test test.o -pthread -lgtest -L/usr/lib

main: main.cpp
	$(CC) $(CC_FLAGS) main.cpp -o main

cuda_tar: cuda_main.cpp \
			cuda.cu \
			FileWriter.cpp \
			FileWriter.h \
			FileWriterTest.cpp \
			cpu_main.cpp \
			makefile
	tar -cf cuda.tar $^
	
clean:
	rm -rf *.o ./cuda ./cpu ./main