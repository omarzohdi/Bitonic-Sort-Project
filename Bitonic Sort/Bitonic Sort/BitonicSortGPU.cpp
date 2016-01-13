//BitonicSortGPU.cpp


/* Ascending: 0, Descending: -1 */
#define DIRECTION 0

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include "OpenCL.h"
#include <CL/cl.h>
#include "BitonicSortGPU.h"

/* 
OCLHandle - Handle containing all the Opencl Necessary objects.
Size - Memory size of the Array
Data - Array to be sorted

The BitonicSort_GPU function starts with 

*/
int * BitonicSort_GPU(OpenCLData * OCLHandle,  int & size, float * data)
{
	int direction;

	OCLHandle->err = clGetKernelWorkGroupInfo(OCLHandle->kernels.kernel_init, OCLHandle->device, CL_KERNEL_WORK_GROUP_SIZE,
	sizeof(OCLHandle->local_size), &OCLHandle->local_size, NULL);

	ISError(OCLHandle->err);

	OCLHandle->local_size = (int)pow(2, floor(log((float)OCLHandle->local_size)/log(2.0f)));
	//Create the for the kernels.
	OCLHandle->dbuffer = clCreateBuffer(OCLHandle->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, data, &OCLHandle->err);
	ISError(OCLHandle->err);

	//set Arguments for the kernel, pass the buffer which holds the data from the array to be sorted
	OCLHandle->err = clSetKernelArg(OCLHandle->kernels.kernel_init, 0, sizeof(cl_mem), &OCLHandle->dbuffer);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_stage_0, 0, sizeof(cl_mem), &OCLHandle->dbuffer);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_stage_n, 0, sizeof(cl_mem), &OCLHandle->dbuffer);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_merge, 0, sizeof(cl_mem), &OCLHandle->dbuffer);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_merge_last, 0, sizeof(cl_mem), &OCLHandle->dbuffer);
	ISError(OCLHandle->err);

	//Calculate the argsize based ont the localsize of the workgroups.
	size_t ArgSize= 8*OCLHandle->local_size*sizeof(float);

	//Set Memory aside for the kernel Arg to be returned later on.
	OCLHandle->err = clSetKernelArg(OCLHandle->kernels.kernel_init, 1, ArgSize, NULL);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_stage_0, 1, ArgSize, NULL);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_stage_n, 1, ArgSize, NULL);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_merge, 1, ArgSize, NULL);
	OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_merge_last, 1, ArgSize, NULL);
	ISError(OCLHandle->err);

	//Enqueue the initial Kernel for sorting the array the parameters passed here are the gl
	OCLHandle->global_size = (size/sizeof(float))/8;
	if(OCLHandle->global_size < OCLHandle->local_size)
		OCLHandle->local_size = OCLHandle->global_size;

	OCLHandle->err = clEnqueueNDRangeKernel(OCLHandle->queue, OCLHandle->kernels.kernel_init, 1, NULL, &OCLHandle->global_size, &OCLHandle->local_size, 0, NULL, NULL); 
	ISError(OCLHandle->err);

	/* Execute further stages */
	int num_stages = OCLHandle->global_size/OCLHandle->local_size;
	for(int hstage = 2; hstage < num_stages; hstage <<= 1) {

		OCLHandle->err = clSetKernelArg(OCLHandle->kernels.kernel_stage_0, 2, sizeof(int), &hstage); //passes the argumetns to the kernel that executes stage zero of the bitonic splits.     
		OCLHandle->err |= clSetKernelArg(OCLHandle->kernels.kernel_stage_n, 3, sizeof(int), &hstage); //passes the arguments to the kernel that execute stage n of the bitonic splits
		ISError(OCLHandle->err);

		//This loop executes the kernel stage n before passing the data to stage zero.
		for( int  stage = hstage; stage > 1; stage >>= 1) {

			OCLHandle->err = clSetKernelArg(OCLHandle->kernels.kernel_stage_n, 2, sizeof(int), &stage); 
			ISError(OCLHandle->err);

			OCLHandle->err = clEnqueueNDRangeKernel(OCLHandle->queue, OCLHandle->kernels.kernel_stage_n, 1, NULL, &OCLHandle->global_size, &OCLHandle->local_size, 0, NULL, NULL); 
			ISError(OCLHandle->err);

		}

		//After executing the previous stages not executes kernel stage 0 --> the arguments were passed before..
		OCLHandle->err = clEnqueueNDRangeKernel(OCLHandle->queue, OCLHandle->kernels.kernel_stage_0, 1, NULL, &OCLHandle->global_size, &OCLHandle->local_size, 0, NULL, NULL); 
		ISError(OCLHandle->err);
	}

	//This sets the sort direction for each Kernel
	direction = DIRECTION;
	OCLHandle->err  = clSetKernelArg(OCLHandle->kernels.kernel_merge, 3, sizeof(int), &direction);
	OCLHandle->err  |= clSetKernelArg(OCLHandle->kernels.kernel_merge_last, 2, sizeof(int), &direction);
	ISError(OCLHandle->err);


	//start Merge the lower stages of tthe Arrays!
	for(int  stage = num_stages; stage > 1; stage >>= 1) {

		OCLHandle->err  = clSetKernelArg(OCLHandle->kernels.kernel_merge, 2, sizeof(int), &stage);
		ISError(OCLHandle->err);


		OCLHandle->err  = clEnqueueNDRangeKernel(OCLHandle->queue, OCLHandle->kernels.kernel_merge, 1, NULL, &OCLHandle->global_size, &OCLHandle->local_size, 0, NULL, NULL); 
		ISError(OCLHandle->err);

	}

	///Merge the final stage///
	OCLHandle->err  = clEnqueueNDRangeKernel(OCLHandle->queue, OCLHandle->kernels.kernel_merge_last, 1, NULL, &OCLHandle->global_size, &OCLHandle->local_size, 0, NULL, NULL); 
	ISError(OCLHandle->err);

	//Gets tje results of the sorting
	OCLHandle->err  = clEnqueueReadBuffer(OCLHandle->queue, OCLHandle->dbuffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
	ISError(OCLHandle->err);

	//CheckResults(direction, data, size/sizeof(float), OCLHandle );

	return 0;
}


void CheckResults(int direction, float * data, int length, OpenCLData * OCLHandle)
{
	
	//Check if the sorting was succesful if it was then do nothing if it wasn't print out that it failed
	//Only used it for debug, it would affect the time otherwise.
	for(int i=1; i<length; i++)
	{
		if(direction == 0) 
			if(data[i] < data[i-1]) {
				std::cout << "Bitonic sort failed!\n";
				break;
			}
		if(direction == -1) 
			if(data[i] > data[i-1]) {
				std::cout << "Bitonic sort failed!\n";
				break;
			}
	}
	
}
