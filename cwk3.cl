// Kernel for matrix transposition.

// from cw spec:
// the i'th row and j'th column has index i*nCols+j,
// which needs to be mapped to j*nRows+i

__kernel
void vectorAdd(__global float *a, int i, int j)
{
    // gets the position of the current element and its 'opposite' to be transposed
    // swaps them
    int gid = get_global_id(0); // (i*nCols+j)

    int buffer = a[gid]
    int inverse = j*((gid - j) / i)+ i
    a[gid] = a[inverse];
    a[inverse] = buffer;
}s