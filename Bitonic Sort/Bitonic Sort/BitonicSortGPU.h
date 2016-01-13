#ifndef BITONICGPU_H
#define BITONICGPU_H

#include "OpenCL.h"

int * BitonicSort_GPU(OpenCLData * OCLHandle,  int & size, float * arr);
void CheckResults(int direction, float * data, int length, OpenCLData * OCLHandle );

#endif
