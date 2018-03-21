#define MAX_WORD_LEN 100
#define WORK_GROUP_SIZE 128

typedef struct word{
    char word[MAX_WORD_LEN];
    int len;
}word_t;


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

__kernel void justsplitdata(const float stepsize, __global const char* val, __global float* out ) {
	int id = get_global_id(0);
	int idx = 0;
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
		if(temp == 0.0) {
			temp = -1.0;
		}

		idx = floor(id / stepsize);
		
		out[idx-1] = temp;
		
	}
}

__kernel void splithistdata(__global const char* val, __global int* hist, __global float *metrics) {
	unsigned int id = get_global_id(0);
	
	if (val[id] == '\n')
	{
		float temp = 0;
		float div = 10.0;

		for (unsigned int i = 5; i > 1; i--) {
			char value = val[id - (i)];
			if (value != ' ' && value != '.')
			{
				temp = (temp * 10) + (value - 48);
			}
		}

		temp = temp / div;
		
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
	}
}


//__kernel void butterflySort(__global const float* val, __global float* outval) {
//	int id = get_global_id(0);
//	int globalSize = get_global_size(0) / 2;
//	int localSize = get_local_size(0);
//
//
//
//
//}

