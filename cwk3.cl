// Kernel for matrix transposition.

// from cw spec:
// the i'th row and j'th column has index i*nCols+j,
// which needs to be mapped to j*nRows+i

__kernel
void Transpose(__global float *a, const int nRows, const int nCols)
{
    // gets the position of the current element and its 'opposite' to be transposed
    // swaps them
    int gid = get_global_id(0); 
    int i = gid / nCols; // row
    int j = gid % nCols; // column

    // only process upper triangle to avoid double swap
    if (i<j && i<nRows && j<nCols)
    {
        int inverse = j*nRows+i;
        int original = i*nCols+j;

        // swap original with opposite
        float buffer = a[original];
        a[original] = a[inverse];
        a[inverse] = buffer;
    }

}