#pragma once

#include <fstream>
#include "Timer.hpp"

//Fast and Efficient File Parser
// Reads the last decimal column of an input file and parses to a vector.
namespace Parse {
	//Convert to a representative data type and push to the destination vector
	void NumericData(char data[], std::vector<int>& dest) {
		dest.push_back(::atoi(data));
	};

	void NumericData(char data[], std::vector<float>& dest) {
		dest.push_back(::atof(data));
	};

	//File Reader/Parser
	template<typename T>
	void FileEOL(std::string file_path, std::vector<T>& destination) {

		//Open input stream to file
		//Seek to the end of the file and get the position of the last char
		std::ifstream input_file(file_path, std::ios::in | std::ios::binary | std::ios::ate);

		//Get Size of File
		std::size_t size = input_file.tellg();

		std::size_t sizeBytes = size * sizeof(int);//size in bytes

		//Seek back to the start of the file
		input_file.seekg(0, std::ios_base::beg);

		//Read into char buffer with size of file
		char * inputElements = new char[size];
		input_file.read(&inputElements[0], size);

		//Close the file
		input_file.close();

		vector<float> temps(size);
		// define size of buffers for OpenCL
		cl::Buffer buffer_val(context, CL_MEM_READ_WRITE, sizeBytes);
		cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeBytes);
		//Copy array to device memory
		cl::Event val_event;
		queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
		// Create kernel instance
		cl::Kernel kernel_splitdata = cl::Kernel(program, "splitdata");
		// Set arguments for kernel (in and out)
		kernel_splitdata.setArg(0, buffer_val);
		kernel_splitdata.setArg(1, buffer_out);
		// Run kernel
		cl::Event prof_splitdata;
		queue.enqueueNDRangeKernel(kernel_splitdata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_splitdata);
		// Retrieve output from OpenCL
		queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, size, &temps[0]);

		long timetaken;
		timetaken = (prof_splitdata.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_splitdata.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
		timetaken += (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time

		std::cout << "Time taken to parse file [ms]:" << timetaken << endl;
		std::cout << "Time taken to run program [ms]:" << timer.End() << endl;

		//For every char if '\n' reached parse the decimal previous
		//for (unsigned int i = 0; i < size; ++i) {
		//	if (inputElements[i] == '\n') {
		//		//Null terminate to support other platforms
		//		char word[7] = { "     \0" };
		//		int j = i, counter = 6;

		//		//Go back to front for efficieny
		//		while (inputElements[j] != ' ')
		//			*(word + counter--) = inputElements[j--];

		//		//Parse data to templated type
		//		Parse::NumericData(word, destination);
		//	}
		//};
	};

	//Wrapper function to record time taken to parse file
	template<typename T>
	void File(std::string file_path, std::vector<T>& destination) {
		ProgramTimer t;
		t.Start();
		Parse::FileEOL(file_path, destination);
		std::cout << "File Parsed in " << t.End() / 1000000 << "ms" << std::endl;
	};

	//Wrapper function when vector not passed by reference, returns a copy.
	template<typename T>
	std::vector<T> File(std::string file_path) {
		std::vector<T> data;
		File(file_path, data);
		return data;
	};
}