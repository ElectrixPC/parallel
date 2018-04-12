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

vector<float> generateBins(string bins) {
	vector<float> mybins(6);
	bins = "," + bins + ",";
	int arrayCounter = 0;
	for (int i = 0; i < bins.size(); ++i) {
		if (bins[i] == ',' && i != (bins.size() - 1)) {
			float myval = 0;
			bool decimal = false;
			bool negative = false;
			for (int j = 0; j < 10; ++j) {
				if (bins[i + j] != ',') {
					if (bins[i + j] == '.') {
						decimal = true;
						continue;
					}
					if (bins[i + j] == '-') {
						negative = true;
						continue;
					}
					if (decimal == true) {
						float val = bins[i + j] - 48;
						float dec = val / 10;
						myval = myval + dec;
						decimal = false;
					}
					else
						myval = (10 * myval) + (bins[i + j] - 48);
				}
				else if (myval != 0.0)
					break;
			}
			if (negative == true)
				myval = myval - (myval * 2);
			mybins[arrayCounter] = myval;
			arrayCounter++;
		}
	}
	return mybins;
}

void parseGetHistMetrics(const char* station, int season, vector<float> mybins, char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	int sizeBytes = size * sizeof(char);
	vector<int> hist(pow(25, 2));
	vector<float> metrics(pow(25, 2));
	cl::Event val_event;
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	cl::Buffer buffer_station(context, CL_MEM_READ_ONLY, sizeof(station));
	cl::Buffer buffer_myhist(context, CL_MEM_READ_ONLY, (sizeof(float) * 6));
	cl::Buffer buffer_mybins(context, CL_MEM_READ_WRITE, 6 * sizeof(float));
	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
	queue.enqueueWriteBuffer(buffer_station, CL_TRUE, 0, sizeof(station), &station[0], NULL, NULL);
	queue.enqueueWriteBuffer(buffer_mybins, CL_TRUE, 0, 6, &mybins[0], NULL, NULL);
	cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	cl::Buffer buffer_metrics(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	//Copy array to device memory
	cl::Event idx_event;
	// Create kernel instance
	cl::Kernel kernel_parsehistdata = cl::Kernel(program, "parsealldata");
	// Set arguments for kernel (in and out)
	kernel_parsehistdata.setArg(0, season);
	kernel_parsehistdata.setArg(1, buffer_station);
	kernel_parsehistdata.setArg(2, buffer_mybins);
	kernel_parsehistdata.setArg(3, buffer_val);
	kernel_parsehistdata.setArg(4, buffer_hist);
	kernel_parsehistdata.setArg(5, buffer_metrics);
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
	std::cout << "\n---------------- Histogram -------------------" << endl;
	std::cout << "Bin 1: " << hist[0] << getHistOutput(hist[0], totalSize) << endl;
	std::cout << "Bin 2: " << hist[1] << getHistOutput(hist[1], totalSize) << endl;
	std::cout << "Bin 3: " << hist[2] << getHistOutput(hist[2], totalSize) << endl;
	std::cout << "Bin 4: " << hist[3] << getHistOutput(hist[3], totalSize) << endl;
	std::cout << "Bin 5: " << hist[4] << getHistOutput(hist[4], totalSize) << endl;
	std::cout << endl;
	std::cout << "Min value: " << metrics[0] << " Max value: " << metrics[1] << endl;
}

void parseGetHist(const char* station, int season, vector<float> mybins, char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	int sizeBytes = size * sizeof(char);
	vector<int> hist(pow(25, 2));
	cl::Event val_event;
	cl::Buffer buffer_station(context, CL_MEM_READ_ONLY, sizeof(station));
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	cl::Buffer buffer_mybins(context, CL_MEM_READ_WRITE, 6 * sizeof(float));
	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
	queue.enqueueWriteBuffer(buffer_station, CL_TRUE, 0, sizeof(station), &station[0], NULL, NULL);
	queue.enqueueWriteBuffer(buffer_mybins, CL_TRUE, 0, sizeof(float) * 6, &mybins[0], NULL, NULL);
	cl::Buffer buffer_hist(context, CL_MEM_READ_WRITE, pow(25, 2) * sizeof(float));
	//Copy array to device memory
	cl::Event idx_event;
	// Create kernel instance
	cl::Kernel kernel_parsehistdata = cl::Kernel(program, "parsehistdata");
	// Set arguments for kernel (in and out)

	kernel_parsehistdata.setArg(0, season);
	kernel_parsehistdata.setArg(1, buffer_station);
	kernel_parsehistdata.setArg(2, buffer_mybins);
	kernel_parsehistdata.setArg(3, buffer_val);
	kernel_parsehistdata.setArg(4, buffer_hist);
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
	float totalSize = (hist[0] + hist[1] + hist[2] + hist[3] + hist[4]);
	std::cout << "\n---------------- Histogram -------------------" << endl;
	std::cout << "Bin 1: " << hist[0] << getHistOutput(hist[0], totalSize) << endl;
	std::cout << "Bin 2: " << hist[1] << getHistOutput(hist[1], totalSize) << endl;
	std::cout << "Bin 3: " << hist[2] << getHistOutput(hist[2], totalSize) << endl;
	std::cout << "Bin 4: " << hist[3] << getHistOutput(hist[3], totalSize) << endl;
	std::cout << "Bin 5: " << hist[4] << getHistOutput(hist[4], totalSize) << endl;
	std::cout << endl;
}

vector<float> justParseData(const char* station, int season, char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {

	vector<unsigned int> minline(1);
	int sizeBytes = size * sizeof(char);
	cl::Buffer buffer_station(context, CL_MEM_READ_ONLY, sizeof(station));
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	cl::Buffer buffer_minline(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));
	cl::Event val_event;

	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
	queue.enqueueWriteBuffer(buffer_station, CL_TRUE, 0, sizeof(station), &station[0], NULL, NULL);
	cl::Kernel kernel_getminline = cl::Kernel(program, "getminline");
	kernel_getminline.setArg(0, buffer_val);
	kernel_getminline.setArg(1, buffer_minline);
	queue.enqueueNDRangeKernel(kernel_getminline, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, NULL);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_minline, CL_TRUE, 0, 1, &minline[0]);

	int lines = size / 5;

	float stepsize = minline[0] - 3;
	std::size_t sizeFLTBytes = (lines * 2.0) * sizeof(float);
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
	kernel_parsedata.setArg(1, season);
	kernel_parsedata.setArg(2, buffer_station);
	kernel_parsedata.setArg(3, buffer_val);
	kernel_parsedata.setArg(4, buffer_out);
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


vector<float> histParseData(const char* station, int season, vector<float> mybins, char* inputElements, int size, cl::Context context, cl::CommandQueue queue, cl::Program program) {

	vector<unsigned int> minline(1);
	int sizeBytes = size * sizeof(char);

	vector<int> hist(pow(25, 2));
	vector<float> metrics(pow(25, 2));
	cl::Buffer buffer_station(context, CL_MEM_READ_ONLY, sizeof(station));
	cl::Buffer buffer_val(context, CL_MEM_READ_ONLY, sizeBytes);
	cl::Buffer buffer_minline(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));
	cl::Buffer buffer_mybins(context, CL_MEM_READ_WRITE, 6 * sizeof(float));
	cl::Event val_event;

	queue.enqueueWriteBuffer(buffer_val, CL_TRUE, 0, sizeBytes, &inputElements[0], NULL, &val_event);
	queue.enqueueWriteBuffer(buffer_station, CL_TRUE, 0, sizeof(station), &station[0], NULL, NULL);
	queue.enqueueWriteBuffer(buffer_mybins, CL_TRUE, 0, sizeof(float) * 6, &mybins[0], NULL, NULL);
	cl::Kernel kernel_getminline = cl::Kernel(program, "getminline");
	kernel_getminline.setArg(0, buffer_val);
	kernel_getminline.setArg(1, buffer_minline);

	queue.enqueueNDRangeKernel(kernel_getminline, cl::NullRange, cl::NDRange(size), cl::NullRange, NULL, NULL);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_minline, CL_TRUE, 0, 1, &minline[0]);


	int lines = size / 5;

	float stepsize = minline[0] - 3;
	std::size_t sizeFLTBytes = (lines * 2.0) * sizeof(float);
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
	kernel_parsehistdata.setArg(1, season);
	kernel_parsehistdata.setArg(2, buffer_station);
	kernel_parsehistdata.setArg(3, buffer_mybins);
	kernel_parsehistdata.setArg(4, buffer_val);
	kernel_parsehistdata.setArg(5, buffer_out);
	kernel_parsehistdata.setArg(6, buffer_hist);
	kernel_parsehistdata.setArg(7, buffer_metrics);
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
	std::cout << "\n---------------- Histogram -------------------" << endl;
	std::cout << "Bin 1: " << hist[0] << getHistOutput(hist[0], totalSize) << endl;
	std::cout << "Bin 2: " << hist[1] << getHistOutput(hist[1], totalSize) << endl;
	std::cout << "Bin 3: " << hist[2] << getHistOutput(hist[2], totalSize) << endl;
	std::cout << "Bin 4: " << hist[3] << getHistOutput(hist[3], totalSize) << endl;
	std::cout << "Bin 5: " << hist[4] << getHistOutput(hist[4], totalSize) << endl;
	std::cout << endl;
	std::cout << "Min value: " << metrics[0] << endl;
	std::cout << "Max value: " << metrics[1] << endl;

	return temps;
}

vector<float> getSorted(vector<float> temps, int lines, cl::Context context, cl::CommandQueue queue, cl::Program program, bool minmax)
{
	ProgramTimer sortTimer;
	sortTimer.Start();
	auto optimal_size = static_cast<std::vector<float>::size_type>(pow(2, ceil(log(temps.size()) / log(2))));
	
	//if the input vector is not a multiple of the local_size
	//insert additional neutral elements (0 for addition) so that the total will not be affected
	if (optimal_size) {
		//create an extra vector with neutral values
		std::vector<float> A_ext(optimal_size - temps.size(), 0.0);
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

	init_bitonic.setArg(0, data_buffer);
	init_bitonic.setArg(1, output_buffer);

	//Create vector to read final values
	vector<float> output(lines);

	unsigned int arrayLength = lines /2;
	unsigned int batch = lines / arrayLength;
	size_t global = batch * lines / 2;
	size_t local = LOCAL_SIZE_LIMIT / 2;
	int dir = 1;
	//Initially call the kernel so that the input buffer can be replaced with the modified/sorted data
	queue.enqueueNDRangeKernel(init_bitonic, cl::NullRange, cl::NDRange(global), cl::NDRange(local), NULL, NULL);
	std::cout << "Time taken to create bitonic sequences [ms]:" << sortTimer.End() << endl;

	for (unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1) {
		for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
			bitonic_merge_global.setArg(0, output_buffer);
			bitonic_merge_global.setArg(1, output_buffer);
			bitonic_merge_global.setArg(2, arrayLength);
			bitonic_merge_global.setArg(3, size);
			bitonic_merge_global.setArg(4, stride);
			bitonic_merge_global.setArg(5, dir);
				
			//printf("starting kernel MergeGlobal %2d out of %d (size %4u stride %4u)\n", run, total, size, stride);
			
			
			queue.enqueueNDRangeKernel(bitonic_merge_global, cl::NullRange, cl::NDRange(global), cl::NDRange(local), NULL, NULL);
		}
	}
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, lines, &output[0]);
	if (minmax == true) {
		int val1 = std::count(temps.begin(), temps.end(), 99.0);
		int val2 = std::count(temps.begin(), temps.end(), 0.0);
		float actualSize = lines - (val1 + val2);
		std::cout << "Min value: " << output[0] << endl;
		std::cout << "Max value: " << output[actualSize-1] << endl;
	}
	std::cout << "Time taken to sort data [ms]:" << sortTimer.End() << endl;
	return output;

}

float getMeanValue(vector<float> temps, int lines, cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	
	int val1 = std::count(temps.begin(), temps.end(), 99.0);
	int val2 = std::count(temps.begin(), temps.end(), 0.0);
	float actualSize = lines - (val1 + val2);
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
	kernel_getmymean.setArg(2, cl::Local(512 * sizeof(float)));//local memory size


	cl::Event prof_meandata;

	queue.enqueueNDRangeKernel(kernel_getmymean, cl::NullRange, cl::NDRange(lines), cl::NullRange, NULL, &prof_meandata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_mymean, CL_TRUE, 0, lines, &mean[0]);

	// TODO get the number of lines to get the mean value

	float meanValue = mean[0] / (actualSize);

	std::cout << "Mean Value: " << meanValue << endl;

	return meanValue;
}

vector<int> getHist(vector<float> mybins, vector<float> temps, int lines, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	// Input is a sorted temps vector

	int val1 = std::count(temps.begin(), temps.end(), 99.0);
	int val2 = std::count(temps.begin(), temps.end(), 0.0);
	float actualSize = lines - (val1 + val2);
	vector<int> myhist(pow(25, 3));
	std::size_t sizeFLTBytes = (lines * 1.8) * sizeof(float);
	cl::Buffer buffer_mytemps(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Buffer buffer_myhist(context, CL_MEM_READ_WRITE, pow(25, 3) * sizeof(float));
	cl::Buffer buffer_mybins(context, CL_MEM_READ_WRITE, 6 * sizeof(float));
	queue.enqueueWriteBuffer(buffer_mybins, CL_TRUE, 0, sizeof(float) * 6, &mybins[0], NULL, NULL);
	queue.enqueueWriteBuffer(buffer_mytemps, CL_TRUE, 0, lines, &temps[0], NULL, NULL);
	cl::Kernel kernel_gethist = cl::Kernel(program, "shitsearch");
	// Set arguments for kernel (in and out)
	kernel_gethist.setArg(0, buffer_mybins);
	kernel_gethist.setArg(1, buffer_mytemps);
	kernel_gethist.setArg(2, buffer_myhist);

	cl::Event prof_histdata;

	queue.enqueueNDRangeKernel(kernel_gethist, cl::NullRange, cl::NDRange(lines), cl::NullRange, NULL, &prof_histdata);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_myhist, CL_TRUE, 0, pow(25, 3), &myhist[0]);

	std::cout << "\n---------------- Histogram -------------------" << endl;
	std::cout << "Bin 1: " << myhist[0] << getHistOutput(myhist[0], actualSize) << endl;
	std::cout << "Bin 2: " << myhist[1] << getHistOutput(myhist[1], actualSize) << endl;
	std::cout << "Bin 3: " << myhist[2] << getHistOutput(myhist[2], actualSize) << endl;
	std::cout << "Bin 4: " << myhist[3] << getHistOutput(myhist[3], actualSize) << endl;
	std::cout << "Bin 5: " << myhist[4] << getHistOutput(myhist[4], actualSize) << endl;
	std::cout << endl;


	return myhist;
}

float getMin(vector<float> temps, cl::Context context, cl::CommandQueue queue, cl::Program program) {

	vector<float> mymin(temps.size());
	std::size_t sizeFLTBytes = (temps.size() * 1.8) * sizeof(float);
	cl::Buffer buffer_mytemps(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Buffer buffer_mymin(context, CL_MEM_READ_WRITE, sizeFLTBytes);

	queue.enqueueWriteBuffer(buffer_mytemps, CL_TRUE, 0, temps.size(), &temps[0], NULL, NULL);
	cl::Kernel kernel_getmin = cl::Kernel(program, "getmin");
	// Set arguments for kernel (in and out)
	kernel_getmin.setArg(0, buffer_mytemps);
	kernel_getmin.setArg(1, buffer_mymin);
	kernel_getmin.setArg(2, cl::Local(128 * sizeof(float)));//local memory size


	queue.enqueueNDRangeKernel(kernel_getmin, cl::NullRange, cl::NDRange(temps.size()), cl::NDRange(256), NULL, NULL);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_mymin, CL_TRUE, 0, temps.size(), &mymin[0]);
	std::cout << "Min value: " << mymin[0] << endl;

	return mymin[0];
}

float getMax(vector<float> temps, cl::Context context, cl::CommandQueue queue, cl::Program program) {

	vector<float> mymax(temps.size());
	std::size_t sizeFLTBytes = (temps.size() * 1.8) * sizeof(float);
	cl::Buffer buffer_mytemps(context, CL_MEM_READ_WRITE, sizeFLTBytes);
	cl::Buffer buffer_mymax(context, CL_MEM_READ_WRITE, sizeFLTBytes);

	queue.enqueueWriteBuffer(buffer_mytemps, CL_TRUE, 0, temps.size(), &temps[0], NULL, NULL);
	cl::Kernel kernel_getmax = cl::Kernel(program, "getmax");
	// Set arguments for kernel (in and out)
	kernel_getmax.setArg(0, buffer_mytemps);
	kernel_getmax.setArg(1, buffer_mymax);
	kernel_getmax.setArg(2, cl::Local(128 * sizeof(float)));//local memory size


	queue.enqueueNDRangeKernel(kernel_getmax, cl::NullRange, cl::NDRange(temps.size()), cl::NDRange(256), NULL, NULL);
	// Retrieve output from OpenCL
	queue.enqueueReadBuffer(buffer_mymax, CL_TRUE, 0, temps.size(), &mymax[0]);
	std::cout << "Max value: " << mymax[0] << endl;

	return mymax[0];
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
		string station = "";
		string seasonc = "";
		int option = 0;
		int output = 0;
		int month = 0;
		std::cout << "MAIN MENU" << endl;
		std::cout << "\nFiltering: " << endl;
		std::cout << "Opt 1: Type name of station you want to see data for, '-' if want all stations\n" << endl;
		std::cin  >> station;
		std::cout << "\nOpt 2: Specify the season, reply using '1' (Winter), '2' (Spring), '3' (Summer), '4' (Winter), 0 for all" << endl;
		std::cout << "If you'd prefer to specify by a specific month type a prefix of 'm' and then the month e.g. m12 for December\n" << endl;
		std::cin >> seasonc;
		if (seasonc.length() == 2)
			month = 90 + (seasonc[1] -48);
		else if (seasonc.length() == 3)
			month = 90 + ((seasonc[1] - 48) * 10) + (seasonc[2] - 48);
		else
			month = seasonc[0];
		int season = month;
		std::cout << "\nDo you want to output the final temp file to a txt? 0 - No 1 - Yes\n" << endl;
		std::cin >> output;
		std::cout << "\nMain Options:" << endl;
		std::cout << "1. - Fastest time to calculate everything in large dataset (Histogram, Min, Max all performed at parse time, mean after)" << endl;
		std::cout << "2. - Fastest time to calculate everything using Sort and Search method in large dataset (Histogram created by sort and searching)\n" << endl;
		std::cout << "Small Tasks:" << endl;
		std::cout << "3. - Just parse data" << endl;
		std::cout << "4. - Just calculate histogram, min and max (Fastest) (at parse (requires parsing))" << endl;
		std::cout << "5. - Just calculate histogram (at parse (requires parsing))" << endl;
		std::cout << "6. - Just calculate histogram (via sort and search)" << endl;
		std::cout << "7. - Just calculate minimum" << endl;
		std::cout << "8. - Just calcualte maximum" << endl;
		std::cout << "9. - Just calculate mean\n" << endl;
		std::cin >> option;

		timer.Start();
		std::ifstream input_file("temp_lincolnshire.txt", std::ios::in | std::ios::binary | std::ios::ate);
		//Get Size of File
		std::size_t size = input_file.tellg();
		input_file.seekg(0, std::ios_base::beg);//Seek back to the start of the file
		//Read into char buffer with size of file
		char * inputElements = new char[size];
		input_file.read(&inputElements[0], size);
		//Close the file
		input_file.close();
		std::cout << "Time taken to open file [ms]:" << timer.End() << endl;
		//TODO ADD SEASON AND LOCATION FILTERS
		vector<float> temps(size);
		vector<float> mybins(6);
		if (option == 1 || option == 2 || option == 4 || option == 5 || option == 6) {
			string bins = "";
			std::cout << "\nPlease enter the 5 bins for the histogram \nDesc:provide 6 values in format 1.0,2.0,3.0,4.0,5.0,6.0 -- will form bins of 1 to 2, 2 to 3, 3 to 4, 4 to 5, 5 to 6.\n" << endl;
			std::cin >> bins;
			mybins = generateBins(bins);
		} 
			
		if (option == 1)
			temps = histParseData(station.c_str(), season, mybins, inputElements, size, context, queue, program);
		if (option == 2 || option == 3 || option >= 6)
			temps = justParseData(station.c_str(), season, inputElements, size, context, queue, program);
		if (option == 1 || option == 2 || option == 9) 
			float mean = getMeanValue(temps, temps.size(), context, queue, program);
		if (option == 2 || option == 6) {
			bool minmax = false;
			if (option == 2)
				minmax = true;
			temps = getSorted(temps, temps.size(), context, queue, program, minmax);
			vector<int> values = getHist(mybins, temps, temps.size(), context, queue, program);
		}
		if (option == 4) 
			parseGetHistMetrics(station.c_str(), season, mybins, inputElements, size, context, queue, program);
		if (option == 5)
			parseGetHist(station.c_str(), season, mybins, inputElements, size, context, queue, program);
		if (option == 7)
			getMin(temps, context, queue, program);
		if (option == 8)
			getMax(temps, context, queue, program);
		
		std::cout << "Total time taken[ms]:" << timer.End() << endl;
		
		if (output == 1) {
			temps.erase(std::remove(temps.begin(), temps.end(), 99.0), temps.end());
			temps.erase(std::remove(temps.begin(), temps.end(), 0.0), temps.end());
			std::replace(temps.begin(), temps.end(), 0.01f, 0.0f);
			std::cout << "Outputting file to output_temps.txt -- Please wait up to 5 minutes" << endl;
			ofstream data_file;
			data_file.open("output_temps.txt", ios::out | ios::binary);
			for (int count = 0; count < temps.size(); count++)
			{
				data_file << temps[count] << '\n';
			}
			data_file.close();
			std::cout << "Successfully outputted file to output_temps.txt" << endl;
		}
		

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	system("pause");

	
	return 0;
}



