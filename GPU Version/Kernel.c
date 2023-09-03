double* Dot(const double *matrix_a, const double *matrix_b,
         const int rows_a, const int cols_a, const int cols_b)
{
    double *result;

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

    return result;
}

double* Transpose(const double *matrix,
                 const int rows, const int cols)
{
    double *output_matrix;

    for (int x = 0; x < cols; x++)
    {
        for (int y = 0; y < rows; y++)
        {
            int index_in = y * cols + x;
            int index_out = x * rows + y;
            
            output_matrix[index_out] = matrix[index_in];
        }
    }

    return output_matrix;
}

double* Add(const double *matrix_a, double *matrix_b,
            const int rows, const int cols)
{
    double *result;

    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = matrix_a[x*cols + y] + matrix_b[x*cols + y];

    return result;
}

double* CDot(const int *number, double *matrix_a,
             const int rows, const int cols)
{
    double *result;

    for (int x = 0; x <= rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = *number * matrix_a[x*cols + y];

    return result;
}

double* sigmoid(const double *matrixSig,
                const int rowsSig, const int colsSig)
{
    double  result [6] = {0};

    for (int x = 0; x < rowsSig; x++)
        for (int y = 0; y < colsSig; y++)
            result[x*colsSig + y] = 1.0/(1.0+exp(-matrixSig[x*colsSig + y]));

    return result;
}

double* FeedForward(const double *input, const double *weights, const double *biases, const int *sizes, const int position)
{
    double* WxA  = Dot(weights, input, sizes[0], 1, sizes[1]);
    double* WApB = Add(WxA, biases, sizes[0], 1);
    double* res  = sigmoid(WApB, sizes[0], 1);

    return res;
}