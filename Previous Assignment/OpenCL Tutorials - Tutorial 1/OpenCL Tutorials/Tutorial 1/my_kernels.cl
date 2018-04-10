#define MAX_WORD_LEN 100
#define WORK_GROUP_SIZE 128

#ifdef LARGE
#define LOCAL_SIZE_LIMIT 256
#define DATA_SIZE 4096
#else
#define LOCAL_SIZE_LIMIT 32
#define DATA_SIZE 128
#endif

typedef struct word{
    char word[MAX_WORD_LEN];
    int len;
}word_t;

inline float GetTemp(int id, __global char* val) {
	float digit;
	float div = 10.0;
	bool negative = false;
	unsigned int idx = 0;
	float temp = 0;

	for (unsigned int i = 6; i > 1; i--) {
		if (i == 6 && val[id - (i)] == '-') {
			negative = true;
			continue;
		}
		else if (i == 6) {
			continue;
		}

		if (val[id - (i)] != ' ' && val[id - (i)] != '.' && i != 6)
		{
			if (val[id - (i)] == '-') {
				negative = true;
			}
			else {
				digit = val[id - (i)] - 48;
				temp = (temp * 10) + digit;
			}
		}
	}

	temp = temp / div;

	if (negative == true) {
		temp = temp - (temp * 2);
	}

	if (temp == 0.0) {
		temp = -99.0;
	}

	return temp;
}

//Function to perform the atomic max
inline void AtomicMax(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = max(prevVal.floatVal, operand);
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


//Function to perform the atomic min
inline void AtomicMin(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = min(prevVal.floatVal, operand);
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

//Function to perform the atomic min
inline void AtomicAdd(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}


__kernel void getminline(__global const char* data, __global unsigned int *minline) {
	unsigned int id = get_global_id(0);
	if (data[id] == '\n') {
		if (minline[0] == 0) {
			minline[0] = 99;
		}
		for (unsigned int i = 36; i > 25; i--) {
			if (data[id + i] == '\n') {
				atomic_min(&minline[0], i);
				break;
			}
		}
	}
}


__kernel void sum(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);


	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid + i] == -99.0) {
				continue;
			}
			else {
				scratch[lid] += scratch[lid + i];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

	}


	if (lid == 0) {
		B[get_group_id(0)] = scratch[lid];

		barrier(CLK_LOCAL_MEM_FENCE);
		if (id == 0) {

			int group_count = get_num_groups(0);
			for (int i = 1; i < group_count; ++i) {
				B[id] += B[i];
			}
		}

	}
}


__kernel void justparsedata(const float stepsize, __global const char* val, __global float* out ) {
	unsigned int id = get_global_id(0);

	if (val[id] == '\n')
	{
		float temp       = GetTemp(id, val);
		unsigned int idx = floor(id / stepsize);
		out[idx-1]       = temp;
	}
}

__kernel void outhistdata(const float stepsize, __global char* val, __global float* out, __global int* hist, __global float *metrics) {
	unsigned int id = get_global_id(0);

	if (val[id] == '\n')
	{
		float temp = GetTemp(id, val);

		if (temp < -10) {
			atomic_inc(&hist[0]);
		}
		else if (temp >= -10 && temp < 0) {
			atomic_inc(&hist[1]);
		}
		else if (temp >= 0 && temp < 5) {
			atomic_inc(&hist[2]);
		}
		else if (temp >= 5 && temp < 15) {
			atomic_inc(&hist[3]);
		}
		else {
			atomic_inc(&hist[4]);
		}

		AtomicMin(&metrics[0], temp);
		AtomicMax(&metrics[1], temp);

		unsigned int idx = floor(id / stepsize);
		out[idx - 1] = temp;

		
	}
}

__kernel void parsealldata(__global const char* val, __global int* hist, __global float *metrics) {
	unsigned int id = get_global_id(0);
	
	if (val[id] == '\n')
	{	
		float temp = GetTemp(id, val);
		
		if (temp < -10) {
			atomic_inc(&hist[0]);
		}
		else if (temp >= -10 && temp < 0) {
			atomic_inc(&hist[1]);
		}
		else if (temp >= 0 && temp < 5) {
			atomic_inc(&hist[2]);
		}
		else if (temp >= 5 && temp < 15) {
			atomic_inc(&hist[3]);
		}
		else {
			atomic_inc(&hist[4]);
		}

		AtomicMin(&metrics[0], temp);
		AtomicMax(&metrics[1], temp);
		AtomicAdd(&metrics[2], temp);
	}
}

__kernel void parsehistdata(__global const char* val, __global int* hist) {
	unsigned int id = get_global_id(0);

	if (val[id] == '\n')
	{
		float temp = GetTemp(id, val);

		if (temp < -10) {
			atomic_inc(&hist[0]);
		}
		else if (temp >= -10 && temp < 0) {
			atomic_inc(&hist[1]);
		}
		else if (temp >= 0 && temp < 5) {
			atomic_inc(&hist[2]);
		}
		else if (temp >= 5 && temp < 15) {
			atomic_inc(&hist[3]);
		}
		else {
			atomic_inc(&hist[4]);
		}
	}
}

inline void ComparatorPrivate(float *A, float *B, int arrowDir)
{
	if ((*A > *B) == arrowDir) {
		float t;
		t = *A;
		*A = *B;
		*B = t;
	}
}

inline void ComparatorLocal(__local float *A, __local float *B, int arrowDir)
{
	if ((*A > *B) == arrowDir) {
		float t;
		t = *A; 
		*A = *B; 
		*B = t;
	}
}

__kernel void init_bitonic(__global float* temps, __global float *output) {
	int N = get_local_size(0);
	int lid = get_local_id(0);
	int id = get_global_id(0);
	__local float A[LOCAL_SIZE_LIMIT];

	async_work_group_copy(A, temps + get_group_id(0)*LOCAL_SIZE_LIMIT, LOCAL_SIZE_LIMIT, 0);

	int comparator = id & ((LOCAL_SIZE_LIMIT / 2) - 1);

	for (int size = 2; size < LOCAL_SIZE_LIMIT; size <<= 1) { 
		int direction = (comparator & (size / 2)) != 0;

		for (int stride = size / 2; stride > 0; stride >>= 1) {
			barrier(CLK_LOCAL_MEM_FENCE);
			int pos = 2 * lid - (lid & (stride - 1));
			/*if (A[pos] == 0.0) {
				A[pos] = 99.0;
			}
			if (A[pos] == -99.0) {
				A[pos] = 0.0;
			}
			if (A[pos + stride] == 0.0) {
				A[pos + stride] = 99.0;
			}
			if (A[pos + stride] == -99.0) {
				A[pos + stride] = 0.0;
			}
*/
			ComparatorLocal(&A[pos], &A[pos + stride], direction);
			
		}

	}

	int direction = (get_group_id(0) & 1);

	for (int stride = LOCAL_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {

		barrier(CLK_LOCAL_MEM_FENCE);

		int pos = 2 * lid - (lid & (stride - 1));
		/*if (A[pos] == 0.0) {
			A[pos] = 99.0;
		}
		if (A[pos] == -99.0) {
			A[pos] = 0.0;
		}
		if (A[pos + stride] == 0.0) {
			A[pos + stride] = 99.0;
		}
		if (A[pos + stride] == -99.0) {
			A[pos + stride] = 0.0;
		}
*/
		ComparatorLocal(&A[pos], &A[pos + stride], direction);


	}


	async_work_group_copy(output + get_group_id(0)* LOCAL_SIZE_LIMIT, A, LOCAL_SIZE_LIMIT, 0);


}

__kernel void bitonic_merge_global(__global float* temps, __global float* output, int arrayLength, int size, int stride, int sortDir) {
	
	int global_comparator = get_global_id(0);
	int comparator = global_comparator  & (arrayLength / 2 - 1);

	int direction = sortDir ^ ((comparator & (size / 2)) != 0);
	int pos = 2 * global_comparator - (global_comparator & (stride - 1));
	float A = temps[pos];
	float B = temps[pos + stride];
	if (A == 0.0) {
		A = 99.0;
	}
	if (A == -99.0) {
		A = 0.0;
	}
	if (B == 0.0) {
		B = 99.0;
	}
	if (B == -99.0) {
		B = 0.0;
	}

	ComparatorPrivate(&A, &B, direction);

	output[pos] = A;
	output[pos + stride] = B;
}

//__kernel void bitonic_merge_local(__global float* temps, __global float* output, int arrayLength, int size, int stride, int sortDir) {
//	int global_comparator = get_global_id(0);
//	int lid = get_local_id(0);
//	int comparator = global_comparator & ((arrayLength / 2) - 1);
//	int direction = sortDir ^ ((comparator & (size / 2)) != 0);
//
//
//	__local float A[LOCAL_SIZE_LIMIT];
//
//	async_work_group_copy(A, temps + get_group_id(0)*LOCAL_SIZE_LIMIT, LOCAL_SIZE_LIMIT, 0);
//
//	for (; stride > 0; stride >>= 1) {
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		int pos = 2 * lid - (lid & (stride - 1));
//		ComparatorLocal(&A[pos], &A[pos + stride], direction);
//	}
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	async_work_group_copy(output + get_group_id(0)* LOCAL_SIZE_LIMIT, A, LOCAL_SIZE_LIMIT, 0);
//
//
//}

__kernel void psearch(__global const float* A, float key, __global int* result) {

	int id = get_global_id(0); 
	int p = get_local_size(0);
	int N = get_global_size(0); 
	int length = N;
	__local int offset;


	if (id < p) {
		offset = 0;

		while (length >= p) {

			length /= p;
			int first = offset + id*length;
			int last = offset + (id + 1)*length - 1;

			if (A[first] <= key && A[last] >= key) {
				offset = first;
			}
			//all threads need to finish before the next iteration
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
		if (!id) {
			if (offset != 0) {
				printf("resultzero: %i", result[0]);
				if (result[0] == 0) {
					result[0] = offset;
					printf("force %i", offset);
				}
				else {
					atomic_min(&result[0], offset);
					printf("offset %i", offset);
				}
			}
		}
	}
}
