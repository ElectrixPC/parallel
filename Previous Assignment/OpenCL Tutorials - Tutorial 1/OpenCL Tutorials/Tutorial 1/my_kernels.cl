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
		out[id] = 1;
	}
}

__kernel void scan_add_atomic(__global int* checked, __global int* checkedidx) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&checkedidx[i], checked[id]);

	//for (int i = 0; i < N; i++) {
	//		if (checked[i] == 0) {
	//			if (i == 0) {
	//				continue;
	//			}
	//			if (checked[i - 1] == 0) {
	//				checkedidx[i] = checkedidx[i-1];
	//			}
	//			if (checked[i - 1] == 1) {
	//				checkedidx[i] = checkedidx[i - 1] + 1;
	//			}
	//
	//		}
	//		if (checked[i] == 1) {
	//			checkedidx[i] = checkedidx[i - 1] + 1;
	//		}
	//}
}

__kernel void splitdata(__global const char* val, __global const int* idx, __global float* out) {
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
		//printf("ID: %i TEMP: %f -- %c %c %c %c", id, temp, val[id-5], val[id-4], val[id-3], val[id-2]);
	 }
	 if(idx[id] != idx[id+1]) { 
		out[idx[id]] = temp;
	}
}

__kernel void splitandsortdata(__global const char* val, __global float* out, __local float* scratch, int merge) {
	int id  = get_global_id(0);
	int lid = get_local_id(0);
    int gid = get_group_id(0);
    int N = get_local_size(0);
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
		//printf("ID: %i TEMP: %f -- %c %c %c %c", id, temp, val[id-5], val[id-4], val[id-3], val[id-2]);
	 }
	out[id] = temp;
}