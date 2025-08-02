// Kernel for matrix transposition.

__kernel
void Transpose(__global const float *input,__global float *output,  const int nRows, const int nCols)
{
    // gets the position of the current element and its 'opposite' to be transposed
    // swaps them
    int gid = get_global_id(0); 
    int i = gid / nCols; // row
    int j = gid % nCols; // column

    // only process upper triangle to avoid double swap
    if (i<nRows && j<nCols)
    {
        int inverse = j*nRows+i;
        int original = i*nCols+j;

        // swap original with opposite
        output[inverse] = input[original];
    }

}