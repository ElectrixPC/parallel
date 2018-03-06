#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Timer.hpp"
#include <CL/cl.hpp>
#include "Utils.h"

using namespace std;

void print_help() { 
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	ProgramTimer timer;
	

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		//display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		timer.Start();
		std::ifstream input_file("temp_lincolnshire_short.txt", std::ios::in | std::ios::binary | std::ios::ate);

		//Get Size of File
		std::size_t size = input_file.tellg();

		std::size_t sizeBytes = size * sizeof(char);//size in bytes
		std::size_t sizeFLTBytes = size * sizeof(float);
												   
		input_file.seekg(0, std::ios_base::beg);//Seek back to the start of the file

		//Read into char buffer with size of file
		char * inputElements = new char[size];
		input_file.read(&inputElements[0], size);

		//Close the file
		input_file.close();

		vector<float> temps(size);
		//vector<int> checked(size);

		//// define size of buffers for OpenCL
		//cl::Buffer buffer_valCheck(context, CL_MEM_READ_WRITE, sizeBytes);
		//cl::Buffer buffer_outCheck(context, CL_MEM_READ_WRITE, sizeIDXBytes);
		////Copy array to device memory
		//cl::Event valCheck_event;
		//queue.enqueueWriteBuffer(buffer_valCheck, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &valCheck_event);


		//// Create kernel instance
		//cl::Kernel kernel_groupdata = cl::Kernel(program, "groupdata");
		//// Set arguments for kernel (in and out)
		//kernel_groupdata.setArg(0, buffer_valCheck);
		//kernel_groupdata.setArg(1, buffer_outCheck);
		//// Run kernel
		//cl::Event prof_groupdata;
		//queue.enqueueNDRangeKernel(kernel_groupdata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_groupdata);
		//// Retrieve output from OpenCL
		//queue.enqueueReadBuffer(buffer_outCheck, CL_TRUE, 0, size, &checked[0]);

		//vector<int> checkedidx(size);

		//// define size of buffers for OpenCL
		//cl::Buffer buffer_outIDX(context, CL_MEM_READ_WRITE, sizeIDXBytes);
		////Copy array to device memory
		//cl::Event valIDX_event;
		//// Create kernel instance
		//cl::Kernel kernel_genidx = cl::Kernel(program, "scan_add_atomic");
		//// Set arguments for kernel (in and out)
		//kernel_genidx.setArg(0, buffer_outCheck);
		//kernel_genidx.setArg(1, buffer_outIDX);
		//// Run kernel
		//cl::Event prof_genidx;
		//queue.enqueueNDRangeKernel(kernel_genidx, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_genidx);
		//// Retrieve output from OpenCL
		//queue.enqueueReadBuffer(buffer_outIDX, CL_TRUE, 0, size, &checkedidx[0]);



		// define size of buffers for OpenCL
		cl::Buffer buffer_val(context, CL_MEM_READ_WRITE, sizeBytes);
		cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeFLTBytes);
		//Copy array to device memory
		cl::Event val_event;
		cl::Event idx_event;
		queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
		// Create kernel instance
		cl::Kernel kernel_splitdata = cl::Kernel(program, "splitdata");
		// Set arguments for kernel (in and out)
		kernel_splitdata.setArg(0, buffer_val);
		//kernel_splitdata.setArg(1, buffer_outIDX);
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
		std::getchar();

		//std::string root(".");
		//string file_path = "temp_lincolnshire.txt";
		//typedef int T;
		//std::vector<T> data = Parse::File<T>(file_path);
		//std::cout << "Size: " << data.size() << ", Last: " << data.back() << '\n' << std::endl;

		//std::ifstream inputFile("temp_lincolnshire.txt", std::ios::in | std::ios::binary | std::ios::ate);
		////Size of File (By calling the current character which is the end)
		//std::size_t elements = inputFile.tellg();
		//size_t size = elements * sizeof(int);//size in bytes
		////Seek back to the start of the file
		//inputFile.seekg(0, std::ios_base::beg);
		////Define char array and then read file into array
		//char * inputElements = new char[elements];
		//int index = 0;
		//while(!inputFile.eof()) {
		//	inputFile.ignore(elements, '\n');

		//	char word[7] = { "     \0" };
		//	int j = inputFile.gcount(), counter = 6;

		//	//Go back to front for efficieny
		//	while (inputElements[j] != ' ')
		//		*(word + counter--) = inputElements[j--];
		//	//inputFile.readsome(inputElements, index+3);
		//	index++;
		//}

		//inputFile.read(&inputElements[0], size);
		////Close file to save memory
		//inputFile.close();
		//// Initialise output array
		//

		///*for (int i = 0; i < inputF.len; i++) {

		//Part 4 - memory allocation
		//host - input

		//ifstream inputFile("temp_lincolnshire.txt");
		//string line;
		//

		//int size = count_line(inputFile);
		//int counter = 0;
		//std::vector<float> temps(size);

		//if (inputFile.is_open())
		//{
		//	while (getline(inputFile, line))
		//	{

		//		temps[counter] = std::stof(find_last(line));
		//		cout << temps[counter] << '\n';
		//	}
		//	inputFile.close();
		//}*/

		//

		//vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
		//vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
		//
		//std::vector<int>A(10000000);
		//std::vector<int>B(10000000);

		//size_t vector_elements = A.size();//number of elements
		//size_t vector_size = A.size()*sizeof(int);//size in bytes

		////host - output
		//vector<int> C(vector_elements);

		////device - buffers
		//cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		//cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		//cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		////Part 5 - device operations

		////5.1 Copy arrays A and B to device memory
		//cl::Event A_event;
		//cl::Event B_event;
		//queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		//queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

		////5.2 Setup and execute the kernel (i.e. device code)
		////cl::Kernel kernel_add = cl::Kernel(program, "add");
		////cl::Kernel kernel_mult = cl::Kernel(program, "multi");
		//cl::Kernel kernel_multiadd = cl::Kernel(program, "multiadd");

		////kernel_mult.setArg(0, buffer_A);
		////kernel_mult.setArg(1, buffer_B);
		////kernel_mult.setArg(2, buffer_C);
		////kernel_add.setArg(0, buffer_C);
		////kernel_add.setArg(1, buffer_B);
		////kernel_add.setArg(2, buffer_C);

		//kernel_multiadd.setArg(0, buffer_A);
		//kernel_multiadd.setArg(1, buffer_B);
		//kernel_multiadd.setArg(2, buffer_C);



		////cl::Event prof_multevent;
		////queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_multevent);
		////cl::Event prof_event;
		////queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//cl::Event prof_event;
		//queue.enqueueNDRangeKernel(kernel_multiadd, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);


		////5.3 Copy the result from device to host
		//queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		//cout << "A = " << A << endl;
		//cout << "B = " << B << endl;
		//cout << "C = " << C << endl;

		////std::cout << "Kernel execution time [ns]:" << prof_multevent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_multevent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		////std::cout << GetFullProfilingInfo(prof_multevent, ProfilingResolution::PROF_US) << endl;


		//std::cout << "Kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		//std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;

		//std::cout << "Transfer time A buffer [ns]:" << A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		//std::cout << GetFullProfilingInfo(A_event, ProfilingResolution::PROF_US) << endl;

		//std::cout << "Transfer time B buffer [ns]:" << B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		//std::cout << GetFullProfilingInfo(B_event, ProfilingResolution::PROF_US) << endl;
	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	system("pause");

	
	return 0;
}
