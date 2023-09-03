__kernel void Dot(__global const double *matrix_a, __global const double *matrix_b, __global float *result,
                                const int rows_a, const int cols_a, const int cols_b)
{
    for (int x = 0; x < rows_a; x++)
    {
        for (int y = 0; y < cols_b; y++)
        {
            float sum = 0.0f;
            for (int k = 0; k < cols_a; k++) {
                sum += matrix_a[y * cols_a + k] * matrix_b[k * cols_b + x];
            }
            
            result[y * cols_b + x] = sum;
        }
    }
}

__kernel void Transpose(__global const double *input_matrix, __global float *output_matrix,
                                const int rows, const int cols)
{    
    for (int x = 0; x < cols; x++)
    {
        for (int y = 0; y < rows; y++)
        {
            int index_in = y * cols + x;
            int index_out = x * rows + y;
            
            output_matrix[index_out] = input_matrix[index_in];
        }
    }
}

__kernel void Add(__global const double *matrix_a, __global double *matrix_b, __global float *result,
                                const int rows, const int cols)
{    
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = matrix_a[x*cols + y] + matrix_b[x*cols + y];
}

__kernel void CDot(__global const int *number, __global double *matrix_a, __global float *result,
                                        const int rows, const int cols)
{    
    for (int x = 0; x <= rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = *number * matrix_a[x*cols + y];
}

__kernel void sigmoid(__global const double *matrix, __global double *result,
                                const int rows, const int cols)
{    
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = 1.0/(1.0+exp(-matrix[x*cols + y]));
}

