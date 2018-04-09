#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#ifdef LARGE
#define LOCAL_SIZE_LIMIT 512
#define DATA_SIZE 4096
#else
#define LOCAL_SIZE_LIMIT 32
#define DATA_SIZE 128
#endif


#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
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

string getHistOutput(int size, int total) {

	string output = "";
	float percent = (float)size / (float)total;
	int actualpercent = percent * 100;
	int sizelength = 10 - to_string(size).length();

	for (int i = 0; i < sizelength; i++) {
		output += " ";
	}
	for (int i = 0; i < actualpercent; i++) {
		output += "=";
	}
	
	return output;
}

void parseGetHistMetrics(char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	int sizeBytes = size * sizeof(char);
	vector<int> hist(pow(25, 2));
	vector<float> metrics(pow(25, 2));
	cl::Event val_event;
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);

	cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	cl::Buffer buffer_metrics(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	//Copy array to device memory
	cl::Event idx_event;
	// Create kernel instance
	cl::Kernel kernel_parsehistdata = cl::Kernel(program, "splithistdata");
	// Set arguments for kernel (in and out)
	kernel_parsehistdata.setArg(0, buffer_val);
	kernel_parsehistdata.setArg(1, buffer_hist);
	kernel_parsehistdata.setArg(2, buffer_metrics);
	// Run kernel
	cl::Event prof_parsehistdata;

	queue.enqueueNDRangeKernel(kernel_parsehistdata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_parsehistdata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_hist, CL_TRUE, 0, pow(25, 2), &hist[0]);
	queue.enqueueReadBuffer(buffer_metrics, CL_TRUE, 0, pow(25, 2), &metrics[0]);
	float totalSize = (hist[0] + hist[1] + hist[2] + hist[3] + hist[4]);
	long timetaken;
	timetaken = (prof_parsehistdata.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_parsehistdata.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
	std::cout << "Time taken to run kernel [ms]:" << timetaken << endl;
	timetaken = (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time
	std::cout << "Time taken to transfer mem [ms]:" << timetaken << endl;
	std::cout << "Histogram Bin 1: " << hist[0] << " Bin 2: " << hist[1] << " Bin 3: " << hist[2] << " Bin 4: " << hist[3] << " Bin 5: " << hist[4] << endl;
	std::cout << "Average (Mean) value: " << metrics[2] / totalSize << " Min value: " << metrics[0] << " Max value: " << metrics[1] << endl;
}

void parseGetHist(char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	int sizeBytes = size * sizeof(char);
	vector<int> hist(pow(25, 2));
	cl::Event val_event;
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);

	cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	//Copy array to device memory
	cl::Event idx_event;
	// Create kernel instance
	cl::Kernel kernel_parsehistdata = cl::Kernel(program, "parsehistdata");
	// Set arguments for kernel (in and out)
	kernel_parsehistdata.setArg(0, buffer_val);
	kernel_parsehistdata.setArg(1, buffer_hist);
	// Run kernel
	cl::Event prof_parsehistdata;

	queue.enqueueNDRangeKernel(kernel_parsehistdata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_parsehistdata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_hist, CL_TRUE, 0, pow(25, 2), &hist[0]);
	long timetaken;
	timetaken = (prof_parsehistdata.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_parsehistdata.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
	std::cout << "Time taken to run kernel [ms]:" << timetaken << endl;
	timetaken = (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time
	std::cout << "Time taken to transfer mem [ms]:" << timetaken << endl;
	std::cout << "Histogram Bin 1: " << hist[0] << " Bin 2: " << hist[1] << " Bin 3: " << hist[2] << " Bin 4: " << hist[3] << " Bin 5: " << hist[4] << endl;
}

vector<float> justParseData(char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {

	vector<unsigned int> minline(1);
	int sizeBytes = size * sizeof(char);
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	cl::Buffer buffer_minline(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));
	cl::Event val_event;

	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);

	cl::Kernel kernel_getminline = cl::Kernel(program, "getminline");
	kernel_getminline.setArg(0, buffer_val);
	kernel_getminline.setArg(1, buffer_minline);
	cl::Event prof_getminline;
	queue.enqueueNDRangeKernel(kernel_getminline, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_getminline);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_minline, CL_TRUE, 0, 1, &minline[0]);


	long timetakenMin;
	timetakenMin = (prof_getminline.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_getminline.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
	timetakenMin += (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time

	std::cout << "Time taken to get line min [ms]:" << timetakenMin << endl;
	// TODO CALCULATE THE MIN LINES TO GET ALL THE VALUES
	int lines = size / 5;

	float stepsize = minline[0] - 2;
	std::size_t sizeFLTBytes = (lines * 1.1) * sizeof(float);
	std::size_t multiSizeFLTBytes = (size) * sizeof(float);
	vector<float> temps(lines);
	// define size of buffers for OpenCL

	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	//Copy array to device memory
	cl::Event idx_event;
	// Create kernel instance
	cl::Kernel kernel_parsedata = cl::Kernel(program, "justparsedata");
	// Set arguments for kernel (in and out)
	kernel_parsedata.setArg(0, stepsize);
	kernel_parsedata.setArg(1, buffer_val);
	kernel_parsedata.setArg(2, buffer_out);
	// Run kernel
	cl::Event prof_parsedata;

	queue.enqueueNDRangeKernel(kernel_parsedata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_parsedata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, lines, &temps[0]);

	long timetakenParse;
	timetakenParse = (prof_parsedata.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_parsedata.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time

	std::cout << "Time taken to parse [ms]:" << timetakenParse << endl;
	
	return temps;
}


vector<float> histParseData(char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {

	vector<unsigned int> minline(1);
	int sizeBytes = size * sizeof(char);

	vector<int> hist(pow(25, 2));
	vector<float> metrics(pow(25, 2));
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	cl::Buffer buffer_minline(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));
	cl::Event val_event;

	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);

	cl::Kernel kernel_getminline = cl::Kernel(program, "getminline");
	kernel_getminline.setArg(0, buffer_val);
	kernel_getminline.setArg(1, buffer_minline);
	cl::Event prof_getminline;
	queue.enqueueNDRangeKernel(kernel_getminline, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_getminline);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_minline, CL_TRUE, 0, 1, &minline[0]);

	long timetakenMin;
	timetakenMin = (prof_getminline.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_getminline.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
	timetakenMin += (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time

	std::cout << "Time taken to get line min [ms]:" << timetakenMin << endl;
	// TODO CALCULATE THE MIN LINES TO GET ALL THE VALUES
	int lines = size / 5;

	float stepsize = minline[0] - 2;
	std::size_t sizeFLTBytes = (lines * 1.1) * sizeof(float);
	vector<float> temps(lines);
	// define size of buffers for OpenCL

	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	cl::Buffer buffer_metrics(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	//Copy array to device memory
	cl::Event idx_event;
	// Create kernel instance
	cl::Kernel kernel_parsehistdata = cl::Kernel(program, "outhistdata");
	// Set arguments for kernel (in and out)
	kernel_parsehistdata.setArg(0, stepsize);
	kernel_parsehistdata.setArg(1, buffer_val);
	kernel_parsehistdata.setArg(2, buffer_out);
	kernel_parsehistdata.setArg(3, buffer_hist);
	kernel_parsehistdata.setArg(4, buffer_metrics);
	// Run kernel
	cl::Event prof_parsehistdata;

	queue.enqueueNDRangeKernel(kernel_parsehistdata, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, &prof_parsehistdata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, lines, &temps[0]);
	queue.enqueueReadBuffer(buffer_hist, CL_TRUE, 0, pow(25, 2), &hist[0]);
	queue.enqueueReadBuffer(buffer_metrics, CL_TRUE, 0, pow(25, 2), &metrics[0]);
	
	float totalSize = (hist[0] + hist[1] + hist[2] + hist[3] + hist[4]);
	long timetakenParse;
	timetakenParse = (prof_parsehistdata.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_parsehistdata.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Kernel execution time
	std::cout << "Time taken to run kernel [ms]:" << timetakenParse << endl;
	timetakenParse = (val_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - val_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000; // Memory transfer time
	std::cout << "Time taken to transfer mem [ms]:" << timetakenParse << endl;
	std::cout << "---------------- Histogram -------------------" << endl;
	std::cout << "Bin 1: " << hist[0] << getHistOutput(hist[0], totalSize) << endl;
	std::cout << "Bin 2: " << hist[1] << getHistOutput(hist[1], totalSize) << endl;
	std::cout << "Bin 3: " << hist[2] << getHistOutput(hist[2], totalSize) << endl;
	std::cout << "Bin 4: " << hist[3] << getHistOutput(hist[3], totalSize) << endl;
	std::cout << "Bin 5: " << hist[4] << getHistOutput(hist[4], totalSize) << endl;
	std::cout << endl;
	std::cout << "Min value: " << metrics[0] << endl;
	std::cout << "Max value: " << metrics[1] << endl;

	temps[lines - 1] = totalSize;

	return temps;
}

vector<float> getSorted(vector<float> temps, int lines, cl::Context context, cl::CommandQueue queue, cl::Program program)
{


	auto optimal_size = static_cast<std::vector<float>::size_type>(pow(2, ceil(log(temps.size()) / log(2))));
	
	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (optimal_size) {
		//create an extra vector with neutral values
		std::vector<int> A_ext(optimal_size - temps.size(), 0);
		//append that extra vector to our input
		temps.insert(temps.end(), A_ext.begin(), A_ext.end());
	}

	lines = temps.size();
	std::size_t sizeFLTBytes = lines * sizeof(float);


	cl::Buffer data_buffer(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, sizeFLTBytes);

	cl::Event mean_event;

	queue.enqueueWriteBuffer(data_buffer, CL_TRUE, 0, lines, &temps[0], NULL, &mean_event);

	//Configure kernels and queue them for execution
	cl::Kernel init_bitonic = cl::Kernel(program, "init_bitonic");
	cl::Kernel bitonic_merge_global = cl::Kernel(program, "bitonic_merge_global");
	cl::Kernel bitonic_merge_local = cl::Kernel(program, "bitonic_merge_local");

	init_bitonic.setArg(0, data_buffer);
	init_bitonic.setArg(1, output_buffer);

	//Create vector to read final values
	vector<float> output(lines);
	cl::Event prof_sortdata;
	
	unsigned int arrayLength = lines;
	unsigned int batch = lines / arrayLength;
	size_t global = batch * lines / 2;
	size_t local = LOCAL_SIZE_LIMIT / 2;
	int dir = 1;
	//Initially call the kernel so that the input buffer can be replaced with the modified/sorted data
	queue.enqueueNDRangeKernel(init_bitonic, cl::NullRange, cl::NDRange(global), cl::NDRange(local), NULL, &prof_sortdata);

	unsigned int total = 0;
	for (unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1) {
		for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
			total++;
		}
	}

	unsigned int run = 0;
	for (unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1) {
		for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
			run++;

			if (stride >= LOCAL_SIZE_LIMIT) {

				bitonic_merge_global.setArg(0, output_buffer);
				bitonic_merge_global.setArg(1, output_buffer);
				bitonic_merge_global.setArg(2, arrayLength);
				bitonic_merge_global.setArg(3, size);
				bitonic_merge_global.setArg(4, stride);
				bitonic_merge_global.setArg(5, dir);
				

				printf("starting kernel MergeGlobal %2d out of %d (size %4u stride %4u)\n", run, total, size, stride);

				bitonic_merge_local.setArg(0, output_buffer);
				queue.enqueueNDRangeKernel(bitonic_merge_global, cl::NullRange, cl::NDRange(global), cl::NDRange(local), NULL, NULL);

			}
			else {
				bitonic_merge_local.setArg(0, output_buffer);
				bitonic_merge_local.setArg(1, output_buffer);
				bitonic_merge_local.setArg(2, arrayLength);
				bitonic_merge_local.setArg(3, size);
				bitonic_merge_local.setArg(4, stride);
				bitonic_merge_local.setArg(5, dir);


				printf("starting kernel MergeLocal  %2d out of %d (size %4u stride %4u)\n", run, total, size, stride);

				queue.enqueueNDRangeKernel(bitonic_merge_local, cl::NullRange, cl::NDRange(global), cl::NDRange(local), NULL, NULL);

			}
		}
	}
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, lines, &output[0]);
	
	return output;
}

float getMeanValue(vector<float> temps, int lines, cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	
	float actualSize = temps[lines - 1];
	temps[lines - 1] = 0.0;
	vector<float> mean(lines);
	std::size_t sizeFLTBytes = lines * sizeof(float);
	cl::Buffer buffer_temps(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Buffer buffer_mymean(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Event mean_event;

	queue.enqueueWriteBuffer(buffer_temps, CL_TRUE, 0, lines, &temps[0], NULL, &mean_event);

	cl::Kernel kernel_getmymean = cl::Kernel(program, "sum");
	// Set arguments for kernel (in and out)
	kernel_getmymean.setArg(0, buffer_temps);
	kernel_getmymean.setArg(1, buffer_mymean);
	kernel_getmymean.setArg(2, cl::Local(256 * sizeof(float)));//local memory size


	cl::Event prof_meandata;

	queue.enqueueNDRangeKernel(kernel_getmymean, cl::NullRange, cl::NDRange(lines), cl::NullRange, NULL, &prof_meandata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_mymean, CL_TRUE, 0, lines, &mean[0]);

	// TODO get the number of lines to get the mean value

	float meanValue = mean[0] / (actualSize);

	std::cout << "Mean Value: " << meanValue << endl;

	return meanValue;
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

		std::cout << "Time taken to open file [ms]:" << timer.End() << endl;

		//vector<unsigned int> minline(1);
		//vector<float> temps = justParseData(inputElements, size, context, queue, program);
		vector<float> temps = histParseData(inputElements, size, context, queue, program);
		std::cout << "Time taken to parse [ms]:" << timer.End() << endl;
		//float mean = getMeanValue(temps, temps.size(), context, queue, program);
		std::cout << "Time taken to get mean [ms]:" << timer.End() << endl;
		temps = getSorted(temps, temps.size(), context, queue, program); 
		//std::cout << "Time taken to parse data & get metrics file [ms]:" << timer.End() << endl;
		
		//parseGetHist(inputElements, size, context, queue, program);
		//parseGetHistMetrics(inputElements, size, context, queue, program);
		std::cout << "Total time taken to complete everything [ms]:" << timer.End() << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	system("pause");

	
	return 0;
}



