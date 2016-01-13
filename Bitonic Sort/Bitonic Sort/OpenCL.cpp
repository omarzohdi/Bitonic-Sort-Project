//OpenCL.cpp

#include "OpenCL.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>


/*Reads from the .cl file and returns the file size and programbuffer for use later
program_size - size of program.
program_buffer - program buffer.
*/
bool getfiledatafromSoruce(char *& program_buffer, 	size_t & program_size)
{
	FILE * program_handle;

	fopen_s (&program_handle, "BitonicSortKernel.cl", "rb");
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);

	program_buffer = (char *)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	return true;
}


/*initalizes the Platform for opencl
OCLHandle - Datastructure containing the Platform object
*/
void InitializePlatform(OpenCLData * OCLHandle)
{
	OCLHandle->err = clGetPlatformIDs(1, &OCLHandle->platform, 0); 
	ISError(OCLHandle->err); 
}

/*initalizes the device for opencl
OCLHandle - Datastructure containing the device object
*/
void InitializeDevice(OpenCLData * OCLHandle)
{
	OCLHandle->err = clGetDeviceIDs(OCLHandle->platform, CL_DEVICE_TYPE_GPU, 1, &OCLHandle->device, 0);
	if (OCLHandle->err == CL_DEVICE_NOT_FOUND)
		OCLHandle->err = clGetDeviceIDs(OCLHandle->platform, CL_DEVICE_TYPE_CPU, 1, &OCLHandle->device, 0);

	ISError(OCLHandle->err);
}
/*initalizes the context for opencl
OCLHandle - Datastructure containing the context object
*/
void InitializeContext(OpenCLData * OCLHandle)
{
	OCLHandle-> context = clCreateContext(0,1,&OCLHandle-> device,0,0,&OCLHandle->err);
	ISError(OCLHandle->err);
}

/*initalizes the program for opencl
if the build fails it throws an assert and prints the log to the console.
OCLHandle - Datastructure containing the program object
*/

void InitializeProgram(OpenCLData * OCLHandle)
{
	char * program_buffer = 0;
	size_t program_size;
	char * log;
	size_t log_size;

	getfiledatafromSoruce(program_buffer, program_size);

	OCLHandle->program = clCreateProgramWithSource(OCLHandle->context, 1, (const char**)&program_buffer, &program_size, &OCLHandle->err);
	ISError(OCLHandle->err);

	free(program_buffer);

	OCLHandle->err = clBuildProgram(OCLHandle->program, 1, &OCLHandle->device, 0, 0, 0);
	
	if (OCLHandle->err > 0)
	{
		clGetProgramBuildInfo(OCLHandle->program, OCLHandle->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		log = new char [log_size + 1];
		log [log_size] = '\0';

		clGetProgramBuildInfo (OCLHandle->program, OCLHandle->device, CL_PROGRAM_BUILD_LOG, log_size, log, 0);
		std::cout << log << std::endl;

		delete log;
	}

	ISError(OCLHandle->err);
}

/*initalizes the kernels for opencl using hardcoded kernel names 
also gets the kernel workgroupinfo and calculates the local size 
OCLHandle - Datastructure containing the kernel datastructure that in turn contains the kernel objects
*/
void InitializeKernels(OpenCLData * OCLHandle)
{
	OCLHandle->kernels.kernel_init = clCreateKernel(OCLHandle->program, "bsort_init", &OCLHandle->err);
	ISError(OCLHandle->err);
	OCLHandle->kernels.kernel_stage_0 = clCreateKernel(OCLHandle->program, "bsort_stage_0", &OCLHandle->err);
	ISError(OCLHandle->err);
	OCLHandle->kernels.kernel_stage_n = clCreateKernel(OCLHandle->program, "bsort_stage_n", &OCLHandle->err);
	ISError(OCLHandle->err);   
	OCLHandle->kernels.kernel_merge = clCreateKernel(OCLHandle->program, "bsort_merge", &OCLHandle->err);
	ISError(OCLHandle->err); 
	OCLHandle->kernels.kernel_merge_last = clCreateKernel(OCLHandle->program, "bsort_merge_last", &OCLHandle->err);
	ISError(OCLHandle->err);

	OCLHandle->err = clGetKernelWorkGroupInfo(OCLHandle->kernels.kernel_init, OCLHandle->device,  
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(OCLHandle->local_size), &OCLHandle->local_size, NULL);

	OCLHandle->local_size = (int)pow(2, floor(log((float)OCLHandle->local_size)/log(2.0f)));
	ISError(OCLHandle->err);
}

/*initalizes the queue for opencl
OCLHandle - Datastructure containing the queue object
*/
void InitializeQueue(OpenCLData * OCLHandle)
{
	OCLHandle->queue = clCreateCommandQueue(OCLHandle->context, OCLHandle->device, 0, &OCLHandle->err);
	ISError(OCLHandle->err);
}

/*initalizes all needed objects for opencl
OCLHandle - Datastructure containg the opencl objects
*/
bool SetupOpenCLEnvironment( OpenCLData * OCLHandle)
{
	InitializePlatform(OCLHandle);
	InitializeDevice(OCLHandle);
	InitializeContext(OCLHandle);
	InitializeProgram(OCLHandle);
	InitializeKernels(OCLHandle);
	InitializeQueue(OCLHandle);

	return true;
}

void CleanupOpenCLEnvironment( OpenCLData * OCLHandle)
{
	clReleaseMemObject(OCLHandle->dbuffer);
	clReleaseKernel(OCLHandle->kernels.kernel_init);
	clReleaseKernel(OCLHandle->kernels.kernel_stage_0);
	clReleaseKernel(OCLHandle->kernels.kernel_stage_n);
	clReleaseKernel(OCLHandle->kernels.kernel_merge);
	clReleaseKernel(OCLHandle->kernels.kernel_merge_last);
	clReleaseCommandQueue(OCLHandle->queue);
	clReleaseProgram(OCLHandle->program);
	clReleaseContext(OCLHandle->context);
}
