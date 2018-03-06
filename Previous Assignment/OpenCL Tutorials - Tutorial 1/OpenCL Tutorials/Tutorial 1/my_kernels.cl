#define MAX_WORD_LEN 100
#define WORK_GROUP_SIZE 128

typedef struct word{
    char word[MAX_WORD_LEN];
    int len;
}word_t;


//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void multi(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

__kernel void multiadd(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = (A[id] * B[id]) + B[id];
}

__kernel void groupdata(__global const char* val, __global int* out) {
	int id = get_global_id(0);
	if (val[id] == '\n') 
	{
		atomic_add(&out[0], 1);
	} 
}

__kernel void linedata(__global const char* val, __global int* out) {
	int id = get_global_id(0);
	if (val[id] == '\n')
	{
		out[id] = out[id-1] + 1;
	}
	else {
		out[id] = out[id - 1];
	}
	printf("ID: %i SIZE: %i", id, out);
}

__kernel void scan_add_atomic(__global int* checked, __global int* checkedidx) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&checkedidx[i], checked[id]);
}

__kernel void splitdata(const float stepsize, __global const char* val,  __global float* out, __global double* metrics) {
	int id  = get_global_id(0);
	float temp = 0;
	float digit;
	float div = 10.0;
	if (val[id] == '\n') 
	{
		for (unsigned int i = 5; i > 1; i--) {
			if(val[id-(i)] != ' ' && val[id-(i)] != '.') 
			{
				digit = val[id-(i)] - 48;
				temp = (temp * 10) + digit;
			}
		}
		temp = temp / div;
		metrics[2] = metrics[2] + temp;
		int idx = floor(id / stepsize);
		out[idx - 1] = temp;
		if(temp < metrics[0]) { 
			metrics[0] = temp;
		}
		if (temp > metrics[1]) {
			metrics[1] = temp;
		}
		//printf("METRICS 0: %f METRICS 1: %f METRICS 2: %f", metrics[0], metrics[1], metrics[2]);
	 }
}
