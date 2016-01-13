#include "opencl.h"
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string>
#include <ctime>
#include <assert.h>

#include "BitonicSortCPU.h"
#include "BitonicSortGPU.h"


#define ELEMENTSIZE  8 // 131072
void printTable(std::string Size, std::string Time, std::string Type);
float * randomArray(int multi, int &size);

/////////////Write Static Algorithm
//////////// Refactor Kernel
//////////// Write PAPER!!

int main ()
{
	OpenCLData OCLHandle;
	std::cout << "Building OpenCL Data Structures....."<< std::endl << std::endl;
	assert (SetupOpenCLEnvironment(&OCLHandle));

	printTable("Size",  "Time - s", "Type");
	std::cout << "-----------------------------------------------------------------\n\n"; 
	for (int i = 1; i< 1280000; i *= 2)		
	{	
		int length = 0;
		float * data = 	randomArray(i, length);
		length /= sizeof(float);

		clock_t start_time, end_time;
		start_time = std::clock();
			Bitonicsort_CPU( data, length ); 
		end_time = std::clock();
		
		delete data;
		printTable (std::to_string(ELEMENTSIZE * i), std::to_string(static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC).append("s"), "CPU");
	}

	std::cout << "\n-----------------------------------------------------------------\n\n"; 
	
	printTable("Size",  "Time - s", "Type");
	std::cout << "-----------------------------------------------------------------\n"; 
	for (int i = 1; i<1280000; i*=2)		
	{	
		int size;
		float * data = 	randomArray(i, size);

		clock_t start_time, end_time;

		start_time = std::clock();
		BitonicSort_GPU(&OCLHandle, size , data); 
		end_time = std::clock();

		//delete data;
		printTable (std::to_string(ELEMENTSIZE * i), std::to_string(static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC).append("s"), "GPU");
	}

	CleanupOpenCLEnvironment(&OCLHandle);
	system("pause");
	return 0;
}

void printTable(std::string Size, std::string Time, std::string Type)
{
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(18) << std::left << Size; 
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(18) << std::left << std::setprecision(3) 
		<< Time;
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << std::setw(18) << std::left << Type;

	std::cout << std::endl;

}

float * randomArray(int multi, int &size )
{
	float * data = new float [ELEMENTSIZE * multi];

	srand((unsigned int)time(NULL));

	for(int j=0; j<ELEMENTSIZE * multi; j++) 
		data[j] = (float)(rand());

	size = (ELEMENTSIZE * multi) * sizeof(float);

	return data;
}