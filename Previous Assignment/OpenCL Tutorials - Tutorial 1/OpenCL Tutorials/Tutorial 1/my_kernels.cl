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

__kernel void justsplitdata(const float stepsize, __global const char* val, __global float* out) {
	int id = get_global_id(0);
	int size = get_global_size(0);
	int idx = 0;
	int prev = 0;
	float temp = 0;
	float digit;
	float div = 10.0;
	if (val[id] == '\n')
	{
		for (unsigned int i = 5; i > 1; i--) {
			if (val[id - (i)] != ' ' && val[id - (i)] != '.')
			{
				digit = val[id - (i)] - 48;
				temp = (temp * 10) + digit;
			}
		}
		temp = temp / div;
		
		for (unsigned int j = 35; j > 28; j--) { 
			if (val[id - (j)] == '\n') {
				prev = id - (j);
				break;
			}
		}
		if (floor(id / stepsize) == floor((prev) / stepsize)) {
			idx = floor((id) / stepsize) + 100;
			printf("ID: %i IDX: %i ", id, idx);
		}
		else {
			idx = floor(id / stepsize);
		}
		
		
		//printf("IDX %i, ID %i, PREV %i, val %f, prev: %f ", idx, id, prev, id / stepsize, prev / stepsize);
		out[idx-1] = temp;
		//printf("%i, %f  prev: %f ", idx, floor(id / stepsize), floor((id - 1) / stepsize));
		//printf("ID: %i TEMP: %f -- %c %c %c %c", id, temp, val[id-5], val[id-4], val[id-3], val[id-2]);
	}
}


__kernel void splitdata(const float stepsize, __global const char* val, __global float* temps, __global long long int* dates, __global double* metrics) {
	int id  = get_global_id(0);
	
	if (val[id] == '\n') 
	{
		float temp = 0;
		float digit;
		int dated;
		long long int date = 0;
		float div = 10.0;
		int idx = floor(id / stepsize);

		for (unsigned int i = 5; i > 1; i--) {
			if(val[id-(i)] != ' ' && val[id-(i)] != '.') 
			{
				if (val[id - (i + 1)] != ' ') {
					digit = val[id - (i)] - 48;
					temp = (temp * 10) + digit;
				}
			}
		}
		temp = temp / div;
		int spacecount = 0;
		for (unsigned int j = 22; j > 5; j--) {
			if (val[id - (j)] != ' ' && spacecount < 6 && (val[id - (j)] - 48) < 10) {
				dated = val[id - (j)] - 48;
				date = (date * 10) + dated;
			}
			else {
				spacecount += 1;
			}
		}
		
		metrics[2] = metrics[2] + temp;
		
		temps[idx - 1] = temp;
		dates[idx] = date;
		
		if(temp < metrics[0]) { 
			metrics[0] = temp;
		}
		if (temp > metrics[1]) {
			metrics[1] = temp;
		}

	 }
}
