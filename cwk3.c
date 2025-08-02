
#include <stdio.h>
#include <stdlib.h>
#include "helper_cwk.h"


int main(int argc, char **argv)
{
    //
    // Parse command line arguments and check they are valid. Handled by a routine in the helper file.
    //
    int nRows, nCols;
    getCmdLineArgs(argc, argv, &nRows, &nCols);

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);

    // Allocate memory for the matrix.
    float *hostMatrix = (float *)malloc(nRows * nCols * sizeof(float));

    // Fill the matrix with random values, and display.
    fillMatrix(hostMatrix, nRows, nCols);
    printf("Original matrix (only top-left shown if too large):\n");
    displayMatrix(hostMatrix, nRows, nCols);

    //beginning of transposition

    // Build the kernel code 'Transpose' contained in the file 'cwk3.cl'. (inspired by VectorAdd from slides 14)
	cl_kernel kernel = compileKernelFromFile( "cwk3.cl", "Transpose", context, device );

    // create buffers on gpu
    cl_mem deviceMatrix = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nRows * nCols * sizeof(float), hostMatrix, &status);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nRows * nCols * sizeof(float), NULL, &status);
	
    //set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &nRows);
    clSetKernelArg(kernel, 3, sizeof(int), &nCols);

    // Set up the global problem size, and the work group size. (inspired by slides 14 VectorAdd)
	size_t indexSpaceSize[1], workGroupSize[1];
	indexSpaceSize[0] = nRows * nCols; // match the command line given size of vector
	workGroupSize [0] = 128;

    // Put the kernel onto the command queue. (inspired by sliodes 14 VectorAdd)
	status = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		printf( "Failure enqueuing kernel: Error %d.\n", status );
		return EXIT_FAILURE;
	}

    // copy result back
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, nRows * nCols * sizeof(float), hostMatrix, 0, NULL, NULL);

    //
    // Display the final result. This assumes that the transposed matrix was copied back to the hostMatrix array
    // (note the arrays are the same total size before and after transposing - nRows * nCols - so there is no risk
    // of accessing unallocated memory).
    //
    printf("Transposed matrix (only top-left shown if too large):\n");
    displayMatrix(hostMatrix, nCols, nRows);

    //
    // Release all resources.
    //
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(deviceMatrix);
    clReleaseMemObject(outputBuffer);
    free(hostMatrix);

    return EXIT_SUCCESS;
}
