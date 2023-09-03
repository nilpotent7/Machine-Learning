#pragma region Math Functions
void Dot(const double matrix_a[], const double matrix_b[], double* result,
         const int rows_a, const int cols_a, const int cols_b)
{
    for (int x = 0; x < cols_b; x++)
        for (int y = 0; y < rows_a; y++)
            for (int k = 0; k < cols_a; k++)
                result[y * cols_b + x] += matrix_a[y * cols_a + k] * matrix_b[k * cols_b + x];
}

void Transpose(const double matrix[], double *result,
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

void Subtract(const double matrix_a[], double matrix_b[], double *result,
            const int rows, const int cols)
{
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = matrix_a[x*cols + y] - matrix_b[x*cols + y];
}

void Add(const double matrix_a[], double matrix_b[], double *result,
            const int rows, const int cols)
{
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = matrix_a[x*cols + y] + matrix_b[x*cols + y];
}

void CDot(const double number, double matrix_a[], double *result,
             const int rows, const int cols)
{
    for (int x = 0; x <= rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = number * matrix_a[x*cols + y];
}

void Sigmoid(const double matrix[], double *result,
                const int rows, const int cols)
{
    for (int x = 0; x < rows; x++)
        for (int y = 0; y < cols; y++)
            result[x*cols + y] = 1.0/(1.0+exp(-matrix[x*cols + y]));
}

void DiagMatrix(const int rows, const double values[], double *result)
{
    result[0] = (1-values[0])*values[0];
    for(int x = 1; x<rows; x++)
        result[x*(rows+1)] = (1-values[x])*values[x];
}

#pragma endregion

void FeedForward(const double *input, const double *weights, const double *biases, double *result, const int *Sizes, const int position, double WxA[], double WApB[])
{
    Dot(weights, input, WxA, Sizes[position+1], Sizes[position], 1);
    Add(WxA, biases, WApB, Sizes[position+1], 1);
    Sigmoid(WApB, result, Sizes[position+1], 1);
}

__kernel void BackProp3Layer(__global const double *Input, const double learningRate, const int dataSize, const int loop, const int loopPerInput, __global const int *Desired_Neurons, __global const int *DesiredN_Index, __global double *Weights, __global double *Biases, __global const int *Sizes, __global double *DebugArray)
{
    #pragma region Filter Common Data
    double W0 [###SIZE:Weights0###] = {0};
    double W1 [###SIZE:Weights1###] = {0};
    double W2 [###SIZE:Weights2###] = {0};

    for(int x = 0; x<###SIZE:Weights0###; x++)
        W0[x] = Weights[x];
    for(int x = 0; x<###SIZE:Weights1###; x++)
        W1[x] = Weights[x+###SIZE:Weights0###];
    for(int x = 0; x<###SIZE:Weights2###; x++)
        W2[x] = Weights[x+###SIZE:Weights0###+###SIZE:Weights1###];


    double B0 [###SIZE:Layer1###] = {0};
    double B1 [###SIZE:Layer2###] = {0};
    double B2 [###SIZE:Layer3###] = {0};

    for(int x = 0; x<###SIZE:Layer1###; x++)
        B0[x] = Biases[x];
    for(int x = 0; x<###SIZE:Layer2###; x++)
        B1[x] = Biases[x+###SIZE:Layer1###];
    for(int x = 0; x<###SIZE:Layer3###; x++)
        B2[x] = Biases[x+###SIZE:Layer1###+###SIZE:Layer2###];

    int S [###SIZE:Layers###] = {0};
    for(int x = 0; x<###SIZE:Layers###; x++)
        S[x] = Sizes[x];
    #pragma endregion

    for(int l = 0; l<loop; l++)
    {
        for(int k = 0; k<dataSize; k++)
        {
            #pragma region Filter Data
            double I   [###SIZE:Layer0###] = {0};
            double DO  [###SIZE:Layer3###] = {0};
            
            for(int x = ###SIZE:Layer0###*k; x<###SIZE:Layer0###*(k+1); x++)
                I[x-###SIZE:Layer0###*k] = Input[x];
            for(int x = 0; x<###SIZE:Layer3###; x++)
                DO[x] = Desired_Neurons[(DesiredN_Index[k]*Sizes[3])+x];

            for(int x = 0; x<4; x++)
                DebugArray[x]   =  I[x];
            for(int x = 0; x<4; x++)
                DebugArray[x+4] = DO[x];

            #pragma endregion

            for(int k2 = 0; k2<loopPerInput; k2++)
            {
                #pragma region Gather Activations
                double A1 [###SIZE:Layer1###] = {0};
                double A2 [###SIZE:Layer2###] = {0};
                double A3 [###SIZE:Layer3###] = {0};

                double WxA1  [###SIZE:Layer1###] = {0};
                double WApB1 [###SIZE:Layer1###] = {0};
                double WxA2  [###SIZE:Layer2###] = {0};
                double WApB2 [###SIZE:Layer2###] = {0};
                double WxA3  [###SIZE:Layer3###] = {0};
                double WApB3 [###SIZE:Layer3###] = {0};

                FeedForward(I,  W0, B0, A1, S, 0, WxA1, WApB1);
                FeedForward(A1, W1, B1, A2, S, 1, WxA2, WApB2);
                FeedForward(A2, W2, B2, A3, S, 2, WxA3, WApB3);

                DebugArray[0] = sizeof(A1);
                DebugArray[1] = sizeof(A1[0]);

                // for(int x = 0; x < sizeof(A1) / sizeof(A1[0]); x++)
                // {
                //     DebugArray
                // }


                double Fn1 [###SIZE:Layer1*Layer1###] = {0};
                double Fn2 [###SIZE:Layer2*Layer2###] = {0};
                double Fn3 [###SIZE:Layer3*Layer3###] = {0};

                DiagMatrix(Sizes[1], A1, Fn1);
                DiagMatrix(Sizes[2], A2, Fn2);
                DiagMatrix(Sizes[3], A3, Fn3);
                
                    
                #pragma endregion

                #pragma region Calculate Sensitivities
                double S3p1  [###SIZE:Layer3*Layer3###] = {0};
                CDot(-2, Fn3, S3p1, Sizes[0], Sizes[0]);

                double Error [###SIZE:Layer3###] = {0};
                Subtract(DO, A3, Error, Sizes[3], 1);

                double S3 [###SIZE:Layer3*Layer3###] = {0};
                Dot(S3p1, Error, S3, Sizes[3], Sizes[3], 1);


                // Sensitivity 2 Calculation
                double W2t [###SIZE:Weights2###] = {0};
                Transpose(W2, W2t, Sizes[3], Sizes[2]);

                double S2p1 [###SIZE:Layer2*Layer3###] = {0};
                Dot(Fn2, W2t, S2p1, Sizes[2], Sizes[2], Sizes[3]);

                double S2 [###SIZE:Layer2###] = {0};
                Dot(S2p1, S3, S2, Sizes[2], Sizes[3], 1);


                // Sensitivity 2 Calculation
                double W1t [###SIZE:Weights1###] = {0};
                Transpose(W1, W1t, Sizes[2], Sizes[1]);

                double S1p1 [###SIZE:Layer1*Layer2###] = {0};
                Dot(Fn1, W1t, S1p1, Sizes[1], Sizes[1], Sizes[2]);

                double S1 [###SIZE:Layer1###] = {0};
                Dot(S1p1, S2, S1, Sizes[1], Sizes[2], 1);

                #pragma endregion

                #pragma region Change Weights and Biases

                // Change Weights and Biases of Layer 2
                double S3L   [###SIZE:Layer3###] = {0};
                CDot(learningRate, S3, S3L, Sizes[3], 1);
                double A2t   [###SIZE:Layer2###] = {0};
                Transpose(A2, A2t, Sizes[2], 1);
                double Diff2 [###SIZE:Weights2###] = {0};
                Dot(S3L, A2t, Diff2, Sizes[3], 1, Sizes[2]);

                double W2n [###SIZE:Weights2###] = {0};
                Subtract(W2, Diff2, W2n, Sizes[3], Sizes[2]);
                double B2n [###SIZE:Layer3###] = {0};
                Subtract(B2, S3L, B2n, Sizes[3], 1);

                // Change Weights and Biases of Layer 1
                double S2L   [###SIZE:Layer2###] = {0};
                CDot(learningRate, S2, S2L, Sizes[2], 1);
                double A1t   [###SIZE:Layer1###] = {0};
                Transpose(A1, A1t, Sizes[1], 1);
                double Diff1 [###SIZE:Weights1###] = {0};
                Dot(S2L, A1t, Diff1, Sizes[2], 1, Sizes[1]);

                double W1n [###SIZE:Weights1###] = {0};
                Subtract(W1, Diff1, W1n, Sizes[2], Sizes[1]);
                double B1n [###SIZE:Layer2###] = {0};
                Subtract(B1, S2L, B1n, Sizes[2], 1);

                // Change Weights and Biases of Layer 1
                double S1L   [###SIZE:Layer1###] = {0};
                CDot(learningRate, S1, S1L, Sizes[1], 1);
                double A0t   [###SIZE:Layer0###] = {0};
                Transpose(I, A0t, Sizes[0], 1);
                double Diff0 [###SIZE:Weights0###] = {0};
                Dot(S1L, A0t, Diff0, Sizes[1], 1, Sizes[0]);

                double W0n [###SIZE:Weights0###] = {0};
                Subtract(W0, Diff0, W0n, Sizes[1], Sizes[0]);
                double B0n [###SIZE:Layer1###] = {0};
                Subtract(B0, S1L, B0n, Sizes[1], 1);

                for(int x = 0; x<###SIZE:Weights0###; x++)
                    W0[x] = W0n[x];
                for(int x = 0; x<###SIZE:Weights1###; x++)
                    W1[x] = W1n[x];
                for(int x = 0; x<###SIZE:Weights2###; x++)
                    W2[x] = W2n[x];

                for(int x = 0; x<###SIZE:Layer1###; x++)
                    B0[x] = B0n[x];
                for(int x = 0; x<###SIZE:Layer2###; x++)
                    B1[x] = B1n[x];
                for(int x = 0; x<###SIZE:Layer3###; x++)
                    B2[x] = B2n[x];

                #pragma endregion
            }
        }
    }

    #pragma region Copy Data to Original Buffers

    // Copy the new Weights back to Original Buffer
    for(int x = 0; x<###SIZE:Weights0###; x++)
        Weights[x] = W0[x];

    for(int x = 0; x<###SIZE:Weights1###; x++)
        Weights[x+###SIZE:Weights0###] = W1[x];

    for(int x = 0; x<###SIZE:Weights2###; x++)
        Weights[x+###SIZE:Weights0###+###SIZE:Weights1###] = W2[x];
    
    // Copy the new Biases back to Original Buffer
    for(int x = 0; x<###SIZE:Layer1###; x++)
        Biases[x] = B0[x];

    for(int x = 0; x<###SIZE:Layer2###; x++)
        Biases[x+###SIZE:Layer1###] = B1[x];

    for(int x = 0; x<###SIZE:Layer3###; x++)
        Biases[x+###SIZE:Layer1###+###SIZE:Layer2###] = B2[x];

    #pragma endregion
}