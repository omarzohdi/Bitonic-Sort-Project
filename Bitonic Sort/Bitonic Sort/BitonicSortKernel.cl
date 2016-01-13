//Macros created to sort and swap vector components they make use of the shuffle and shuffle2
//Kernel functions to resolve the operations quickly.
//in the comparison we XOR with dir to get a 1 or 0 instead of -1 and 0.
//Inputs in Sort are the array and the direcion.
//Inputs in Swap are the two arrays to compare and the direction
#define VECTOR_SORT(input, dir)                                   \
   comp = input < shuffle(input, mask2) ^ dir;                    \
   input = shuffle(input, as_uint4(comp * 2 + add2));             \
   comp = input < shuffle(input, mask1) ^ dir;                    \
   input = shuffle(input, as_uint4(comp + add1));                 \

#define VECTOR_SWAP(input1, input2, dir)                          \
   temp = input1;                                                 \
   comp = (input1 < input2 ^ dir) * 4 + add3;                     \
   input1 = shuffle2(input1, input2, as_uint4(comp));             \
   input2 = shuffle2(input2, temp, as_uint4(comp));               \

//this one performs the initial sort, makes the array into bitonic sequence.
__kernel void bsort_init(__global float4 *g_data, __local float4 *l_data) {
   int dir;
   uint id;
	uint global_start;
	uint size;
	uint stride;

   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1	= (uint4)(1, 0, 3, 2);
   uint4 mask2	= (uint4)(2, 3, 0, 1);
   uint4 mask3	= (uint4)(3, 2, 1, 0);

   int4 add1	= (int4)(1, 1, 3, 3);
   int4 add2	= (int4)(2, 3, 2, 3);
   int4 add3	= (int4)(1, 2, 2, 3);

   id = get_local_id(0) * 2;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   input1 = g_data[global_start]; 
   input2 = g_data[global_start+1];

   //Sorts the first input in ascending order, using the mask and the adds
   //along with the shuffle functions.
   comp = input1 < shuffle(input1, mask1);
   input1 = shuffle(input1, as_uint4(comp + add1));
   comp = input1 < shuffle(input1, mask2);
   input1 = shuffle(input1, as_uint4(comp * 2 + add2));
   comp = input1 < shuffle(input1, mask3);
   input1 = shuffle(input1, as_uint4(comp + add3));

   //Does the same but in Descending order.
   comp = input2 > shuffle(input2, mask1);
   input2 = shuffle(input2, as_uint4(comp + add1));
   comp = input2 > shuffle(input2, mask2);
   input2 = shuffle(input2, as_uint4(comp * 2 + add2));
   comp = input2 > shuffle(input2, mask3);
   input2 = shuffle(input2, as_uint4(comp + add3));     

   //Swap the corrsponding elements in input1 and input2
   add3 = (int4)(4, 5, 6, 7);
   dir = get_local_id(0) % 2 * -1;
   temp = input1;

   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));

   //Sort the data using the local memory of the workitem.
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);


   l_data[id] = input1;
   l_data[id+1] = input2;

   //Start the creation of the bitonic sequence!
   for(size = 2; size < get_local_size(0); size <<= 1) 
   {
      dir = (get_local_id(0)/size & 1) * -1;

      for(stride = size; stride > 1; stride >>= 1) 
	  {
         barrier(CLK_LOCAL_MEM_FENCE);
         id = get_local_id(0) + (get_local_id(0)/stride)*stride;
         VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
      }

      barrier(CLK_LOCAL_MEM_FENCE); //<--- barrier is used to sync threads in the workgroup! keep that in mind! 
      id = get_local_id(0) * 2;
      input1 = l_data[id]; input2 = l_data[id+1];
      temp = input1;
      comp = (input1 < input2 ^ dir) * 4 + add3;
      input1 = shuffle2(input1, input2, as_uint4(comp));
      input2 = shuffle2(input2, temp, as_uint4(comp));
      VECTOR_SORT(input1, dir);
      VECTOR_SORT(input2, dir);
      l_data[id] = input1;
      l_data[id+1] = input2;
   }
	
   ///The array at this point should be a bitnoic sequence now we deal with the Bitonic sorting itself!
   //Needs to be merged back into one.
   dir = (get_group_id(0) % 2) * -1;
   for(stride = get_local_size(0); stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE); 
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   //Final sort for the entire array!
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   
   temp = input1; //<--- I don't need this remember to remove.
   comp = (input1 < input2 ^ dir) * 4 + add3;
   
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);

   g_data[global_start] = input1;
   g_data[global_start+1] = input2;
}

// Perform lowest stage of the bitonic sort 
__kernel void bsort_stage_0(__global float4 *g_data, __local float4 *l_data, 
                            uint high_stage) {

   int dir;
   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   //Lookup the data in the global memory find the location
   //use the group_id and high_stage to calc the dir
   id = get_local_id(0);
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   /* Perform initial swap */
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];

   comp = (input1 < input2 ^ dir) * 4 + add3;

   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   /* Perform bitonic merge */
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform final sort */
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];
   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;
   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));
   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);

   /* Store output in global memory */
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}

///Performs a bitonic sort for succssive stages n
__kernel void bsort_stage_n(__global float4 *g_data, __local float4 *l_data, 
                            uint stage, uint high_stage) {

   int dir;
   float4 input1, input2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   //Lookup the data in the global memory find the location
   //use the group_id and high_stage to calc the dir
   dir = (get_group_id(0)/high_stage & 1) * -1;
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   ///perform the swap
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];

   comp = (input1 < input2 ^ dir) * 4 + add;

   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

///USe this one to sort the bitonic set! 
//(same function as the bsort_stage_n only difference is the dir here is passed to the fuction while the
//previous function was dynamic
__kernel void bsort_merge(__global float4 *g_data, __local float4 *l_data, uint stage, int dir) {

   float4 input1, input2;
   int4 comp, add;
   uint global_start, global_offset;

   add = (int4)(4, 5, 6, 7);

   //Lookup the data in the global memory find the location
   global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
   global_offset = stage * get_local_size(0);

   ///perform the swap
   input1 = g_data[global_start];
   input2 = g_data[global_start + global_offset];
   
   comp = (input1 < input2 ^ dir) * 4 + add;

   g_data[global_start] = shuffle2(input1, input2, as_uint4(comp));
   g_data[global_start + global_offset] = shuffle2(input2, input1, as_uint4(comp));
}

//Last Step Merge everything!
__kernel void bsort_merge_last(__global float4 *g_data, __local float4 *l_data, int dir) {

	///Same old procedure
   uint id, global_start, stride;
   float4 input1, input2, temp;
   int4 comp;

   uint4 mask1 = (uint4)(1, 0, 3, 2);
   uint4 mask2 = (uint4)(2, 3, 0, 1);
   uint4 mask3 = (uint4)(3, 2, 1, 0);

   int4 add1 = (int4)(1, 1, 3, 3);
   int4 add2 = (int4)(2, 3, 2, 3);
   int4 add3 = (int4)(4, 5, 6, 7);

   //Find the correct data in the global memory!
   id = get_local_id(0);
   global_start = get_group_id(0) * get_local_size(0) * 2 + id;

   //perform the swap between the two locations just found...
   input1 = g_data[global_start];
   input2 = g_data[global_start + get_local_size(0)];

   comp = (input1 < input2 ^ dir) * 4 + add3;

   l_data[id] = shuffle2(input1, input2, as_uint4(comp));
   l_data[id + get_local_size(0)] = shuffle2(input2, input1, as_uint4(comp));

   /// all that is left is to firs merge everything up!
   for(stride = get_local_size(0)/2; stride > 1; stride >>= 1) 
   {
      barrier(CLK_LOCAL_MEM_FENCE); ///Synch
      id = get_local_id(0) + (get_local_id(0)/stride)*stride;
      VECTOR_SWAP(l_data[id], l_data[id + stride], dir)
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   //Last Sort
   id = get_local_id(0) * 2;
   input1 = l_data[id]; input2 = l_data[id+1];

   temp = input1;
   comp = (input1 < input2 ^ dir) * 4 + add3;

   input1 = shuffle2(input1, input2, as_uint4(comp));
   input2 = shuffle2(input2, temp, as_uint4(comp));

   VECTOR_SORT(input1, dir);
   VECTOR_SORT(input2, dir);

   //Store the final results back into the global memeory!
   g_data[global_start + get_local_id(0)] = input1;
   g_data[global_start + get_local_id(0) + 1] = input2;
}
