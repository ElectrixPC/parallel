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
		std::ifstream input_file("temp_lincolnshire.txt", std::ios::in | std::ios::binary | std::ios::ate);

		//Get Size of File
		std::size_t size = input_file.tellg();

		std::size_t sizeBytes = size * sizeof(char);//size in bytes
		
		std::size_t sizeIDXBytes = size * sizeof(int);
		input_file.seekg(0, std::ios_base::beg);//Seek back to the start of the file

		//Read into char buffer with size of file
		char * inputElements = new char[size];
		input_file.read(&inputElements[0], size);

		//Close the file
		input_file.close();

		
		
		//vector<int> lines(size);

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
		//queue.enqueueReadBuffer(buffer_outCheck, CL_TRUE, 0, size, &lines[0]);

		// do this serial as the cost to transfer the data is higher


		int currmin = 99999;
		int prev = 0;
		for (int i = 0; i < size; i++) {
			prev++;
			if (inputElements[i] == '\n') {
				if (currmin > prev) {
					currmin = prev;
				}
				prev = 0;;
			}
		}
		int lines = size / currmin;
		vector<int> metrics(pow(25, 2));
		float stepsize = currmin - 2;
		//vector<long long int> dates(lines);
		//std::size_t sizeLINTBytes = size * sizeof(long long int);
		std::size_t sizeFLTBytes = (lines * 1.1) * sizeof(float);
		vector<float> temps(lines);
		// define size of buffers for OpenCL
		cl::Buffer buffer_val(context, CL_MEM_READ_WRITE, sizeBytes);
		cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeFLTBytes);
		//cl::Buffer buffer_dates(context, CL_MEM_READ_WRITE, sizeLINTBytes);
		cl::Buffer buffer_metrics(context, CL_MEM_READ_WRITE, pow(25,2)*sizeof(int));
		//Copy array to device memory
		cl::Event val_event;
		cl::Event idx_event;
		queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
		// Create kernel instance
		cl::Kernel kernel_splitdata = cl::Kernel(program, "splithistdata");
		// Set arguments for kernel (in and out)
		kernel_splitdata.setArg(0, stepsize);
		kernel_splitdata.setArg(1, buffer_val);
		kernel_splitdata.setArg(2, buffer_out);
		//kernel_splitdata.setArg(3, buffer_dates);
		kernel_splitdata.setArg(3, buffer_metrics);
		// Run kernel
		cl::Event prof_splitdata;
		queue.enqueueNDRangeKernel(kernel_splitdata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_splitdata);
		// Retrieve output from OpenCL
		queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, lines, &temps[0]);
		//queue.enqueueReadBuffer(buffer_dates, CL_TRUE, 0, lines[0], &dates[0]);
		queue.enqueueReadBuffer(buffer_metrics, CL_TRUE, 0, pow(25,2), &metrics[0]);


		std::cout << "Time taken to run program [ms]:" << timer.End() << endl;


		vector<float> mymax(lines);
		vector<float> mymin(lines);

		cl::Event max_event;
		cl::Event min_event;

		cl::Buffer buffer_max(context, CL_MEM_READ_WRITE, sizeFLTBytes);
		cl::Buffer buffer_min(context, CL_MEM_READ_WRITE, sizeFLTBytes);
		cl::Kernel kernel_mymax = cl::Kernel(program, "mymax");
		cl::Kernel kernel_mymin = cl::Kernel(program, "mymin");
		kernel_mymax.setArg(0, buffer_out);
		kernel_mymax.setArg(1, buffer_max);
		cl::Event prof_mymax;
		queue.enqueueNDRangeKernel(kernel_mymax, cl::NullRange, cl::NDRange(lines), cl::NullRange, NULL, &prof_mymax);
		queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0, lines, &mymax[0]);

		kernel_mymin.setArg(0, buffer_out);
		kernel_mymin.setArg(1, buffer_min);

		cl::Event prof_mymin;
		queue.enqueueNDRangeKernel(kernel_mymin, cl::NullRange, cl::NDRange(lines), cl::NullRange, NULL, &prof_mymin);
		queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0, lines, &mymin[0]);

		long timetaken;
		timetaken = (prof_splitdata.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_splitdata.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
		timetaken += (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time

		std::cout << "Time taken to parse file [ms]:" << timetaken << endl;
		std::cout << "Time taken to run program [ms]:" << timer.End() << endl;
		std::getchar();

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	system("pause");

	
	return 0;
}
