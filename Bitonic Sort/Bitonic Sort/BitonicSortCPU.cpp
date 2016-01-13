//BitonicSortCPU.cpp

#include "BitonicSortCPU.h"
#include <time.h>
#include <assert.h>
#include <iostream>

#define ASCENDING true
#define DESCENDING false

void RecursiveSort(const int lo, const int n, float * a, const bool dir);
void RecursiveMerge(const int lo, const int n, float * a, const bool dir);

/* a - array to be sorted 
length - length of the array

This function takes on the array and legnth of 
the array and setups the recursive sort procedure */

void Bitonicsort_CPU(float * a, const int length)
{
	RecursiveSort(0, length, a, ASCENDING);
}

/*
lo - start index of subsection of the array
n - length of subsection of the array
a - array to be sorted
dir - direction of sort (ascending - descending)

RecursiveSort and RecursiveMerge
Recursively moves to the lower stage of the arrya 
sorts each half of the array to be a monotonic 
sequence and then merges it recursively into one

*/

void RecursiveSort(const int lo, const int n, float * a, const bool dir)
{
    if (n>1)
    {
        int m=n/2;
        RecursiveSort(lo, m,   a, ASCENDING);
        RecursiveSort(lo+m, m, a, DESCENDING);
        RecursiveMerge(lo, n,  a, dir);
    }
}

void RecursiveMerge(const int lo, const int n, float * a, const bool dir)
{
    if (n>1)
    {
        int m=n/2;
        for (int i=lo; i<lo+m; i++)
		{
			if (dir  == ( a[i] > a[i+m] )) 
			{
				float  t=a[i];
				a[i]=a[i+m];
				a[i+m]=t;
			}
		}
        RecursiveMerge(lo, m, a, dir);
        RecursiveMerge(lo+m, m, a, dir);
    }
}
