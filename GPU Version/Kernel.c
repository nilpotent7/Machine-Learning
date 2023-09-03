void Dot(const __global double *matrix_a, const __global double *matrix_b, __global double *result,
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

void Transpose(const __global double *matrix, __global double *result,
                 const int rows, const int cols)
{
    for (int x = 0; x < cols; x++)
    {
        for (int y = 0; y < rows; y++)
        {
            int index_in = y * cols + x;
            int index_out = x * rows + y;
            
            result[index_out] = matrix[index_in];
        }
    }
}

void Add(const __global double *matrix_a, __global double *matrix_b, __global double *result,
            const int rows, const int cols)
{
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = matrix_a[x*cols + y] + matrix_b[x*cols + y];
}

void CDot(const int *number, __global double *matrix_a, __global double *result,
             const int rows, const int cols)
{

    for (int x = 0; x <= rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = *number * matrix_a[x*cols + y];
}

void sigmoid(__global const double *matrixSig, __global double *result,
             const int rowsSig, const int colsSig)
{
    for (int x = 0; x < rowsSig; x++)
        for (int y = 0; y < colsSig; y++)
            result[x*colsSig + y] = 1.0/(1.0+exp(-matrixSig[x*colsSig + y]));
}

double* FeedForward(__global const double *actives, __global const double *weights, __global const double *biases, const int *sizes)
{
    __global double* WxA  = Dot(weights, actives, sizes[0], 1, sizes[1]);
    __global double* WApB = Add(WxA, biases, sizes[0], 1);
    __global double* res  = sigmoid(WApB, sizes[0], 1);
}