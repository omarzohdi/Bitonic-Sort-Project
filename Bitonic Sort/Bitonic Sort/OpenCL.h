#ifndef NQUEEN_GPU_H
#define NQUEEN_GPU_H

#include "CL\cl.h"
#include <assert.h>

///Kernel Collection holds the cl_kernel objects for use later.
struct KernelCollection
{
	cl_kernel kernel_init;
	cl_kernel kernel_stage_0;
	cl_kernel kernel_stage_n;
	cl_kernel kernel_merge;
    cl_kernel kernel_merge_last;
};

///OpenCL datastructure to keep all of the needed buffers and handles for later.
struct OpenCLData
{
	cl_platform_id platform;
	cl_device_id  device;
	cl_context  context;
	cl_program program;
	KernelCollection kernels;
	cl_command_queue queue;
	cl_mem dbuffer;
	cl_int err;
	cl_uint num_kernels;
	size_t local_size, global_size;
};

///Checks if there's an error with the operation and throws an assert in case there is.
inline void ISError(int err){ assert (err < 0 ? false : true); }
bool SetupOpenCLEnvironment(OpenCLData * OCLHandle);

bool getfiledatafromSoruce(char *& program_buffer, 	size_t & program_size);
void InitializePlatform(OpenCLData * OCLHandle);
void InitializeDevice(OpenCLData * OCLHandle);
void InitializeContext(OpenCLData * OCLHandle);
void InitializeProgram(OpenCLData * OCLHandle);
void InitializeKernels(OpenCLData * OCLHandle);
void InitializeBuffersAndArguments(OpenCLData * OCLHandle);
void InitializeQueue(OpenCLData * OCLHandle);
void CleanupOpenCLEnvironment( OpenCLData * OCLHandle);


#endif
