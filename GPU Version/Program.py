import os
import time
import random
import numpy as np
import pyopencl as cl

#region Specific Function Debuggers

def MulitiplyMatrix(program, context, queue):
    matrix_a_rows = 2
    matrix_a_cols = 2
    matrix_b_rows = 2
    matrix_b_cols = 2

    matrix_a = np.array([7, 2, 6, 8]).reshape(2,2).astype(np.double)
    matrix_b = np.array([7, 2, 6, 8]).reshape(1,1).astype(np.double)

    buffer_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_a)
    buffer_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_b)
    buffer_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix_a.nbytes)

    program.matrix_constant_multiply(queue, (1,1), None, buffer_b, buffer_a, buffer_result,
                            np.int32(matrix_a_rows), np.int32(matrix_a_cols))

    result = np.empty((matrix_a_rows, matrix_b_cols), dtype=np.double)
    cl.enqueue_copy(queue, result, buffer_result).wait()

    print("Matrix A:")
    print(matrix_a)
    print("\nMatrix B:")
    print(matrix_b)
    print("\nResult:")
    print(result)

def TransposeMatrix(program, context, queue):
    matrix_rows = 7
    matrix_cols = 5

    matrix = np.random.rand(matrix_rows, matrix_cols).astype(np.double)
    buffer_matrix = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)

    transposed_matrix = np.empty((matrix_cols, matrix_rows), dtype=np.double)
    buffer_transposed = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, transposed_matrix.nbytes)

    program.matrix_transpose(queue, (1, 1), None, buffer_matrix, buffer_transposed,
                            np.int32(matrix_rows), np.int32(matrix_cols))

    cl.enqueue_copy(queue, transposed_matrix, buffer_transposed).wait()

    print("Original Matrix:")
    print(matrix)
    print("\nTransposed Matrix:")
    print(transposed_matrix)

def MatrixAddition(program, context, queue):
    matrix_rows = 4
    matrix_cols = 2

    matrix_a = np.array([72, 12, 46, 51, 12, 46, 72, 51]).reshape(4,2).astype(np.double)
    matrix_b = np.array([77, 62, 76, 28, 24, 22, 32, 12]).reshape(4,2).astype(np.double)
    buffer_matrix_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_a)
    buffer_matrix_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_b)

    result = np.empty((matrix_rows, matrix_cols), dtype=np.double)
    buffer_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result.nbytes)

    program.matrix_addition(queue, (1, 1), None, buffer_matrix_a, buffer_matrix_b, buffer_result,
                            np.int32(matrix_rows), np.int32(matrix_cols))

    cl.enqueue_copy(queue, result, buffer_result).wait()

    print(matrix_a)
    print(matrix_b)
    print(result)

def MulitiplyMatrixByConstant(program, context, queue):
    matrix_a_rows = 8
    matrix_a_cols = 17

    matrix_a = np.random.rand(matrix_a_rows, matrix_a_cols).astype(np.double)
    constant = np.int32(-2)

    buffer_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_a)
    buffer_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=constant)
    buffer_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix_a.nbytes)

    program.matrix_constant_multiply(queue, (1,1), None, buffer_b, buffer_a, buffer_result,
                            np.int32(matrix_a_rows), np.int32(matrix_a_cols))

    result = np.empty((matrix_a_rows, matrix_a_cols), dtype=np.double)
    cl.enqueue_copy(queue, result, buffer_result).wait()

    print("\nResult:")
    print(result)

def Sigmoid(program, context, queue):
    matrix_rows = 4
    matrix_cols = 2

    matrix_a = np.array([7, 12, -4, 5, 2, 6, 7, -5]).reshape(4,2).astype(np.double)
    buffer_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix_a)
    buffer_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, matrix_a.nbytes)
    program.sigmoid2(queue, (1,1), None, buffer_a, buffer_result, np.int32(matrix_rows), np.int32(matrix_cols))

    #result = np.array([7, 12, -4, 5, 2, 6, 7, -5]).reshape(4,2).astype(np.double)
    result = np.empty((4,2), dtype=np.double)
    cl.enqueue_copy(queue, result, buffer_result).wait()

    print("\nResult:")
    print(result)

#endregion

#region External Functions and Network Class
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.position = 0

    def feedforward(self, a):
        z = sigmoid(np.dot(self.weights[self.position], a) + self.biases[self.position])
        # if self.position == 2:
        #     print(np.dot(self.weights[self.position], a))
        self.position += 1
        return z

    def evaluate(self, a):
        for b, w in zip(self.biases, self.weights):
            a = np.around(sigmoid(np.dot(w, a)+b), 2)
        return a
    
    def Backpropagate(self, input, desired_output, learning_rate):
        self.position = 0
        a1  = self.feedforward (input)
        a2  = self.feedforward (a1)
        a3  = self.evaluate    (input)
        
        a1r = a1.reshape(self.sizes[1])
        a2r = a2.reshape(self.sizes[2])
        a3r = a3.reshape(self.sizes[3])
        
        Fn1 = np.diag(np.full(self.sizes[1],(1-a1r)*a1r))
        Fn2 = np.diag(np.full(self.sizes[2],(1-a2r)*a2r))
        Fn3 = np.diag(np.full(self.sizes[3],(1-a3r)*a3r))

        s3 = np.dot(np.dot(-2, Fn3), CalculateError(a3, desired_output))
        s2 = np.dot(np.dot(Fn2, np.transpose(self.weights[2])), s3)
        s1 = np.dot(np.dot(Fn1, np.transpose(self.weights[1])), s2)

        self.weights[2] = self.weights[2] - np.dot(np.dot(learning_rate, s3), np.transpose(a2))
        self.weights[1] = self.weights[1] - np.dot(np.dot(learning_rate, s2), np.transpose(a1))
        self.weights[0] = self.weights[0] - np.dot(np.dot(learning_rate, s1), np.transpose(input))
        self.biases[2]  = self.biases[2]  - np.dot(learning_rate, s3)
        self.biases[1]  = self.biases[1]  - np.dot(learning_rate, s2)
        self.biases[0]  = self.biases[0]  - np.dot(learning_rate, s1)

        return (self.weights, self.biases)

    def SaveData(self, path):
        print("Weights")
        print(self.weights)
        print("Biases")
        print(self.biases)
        for k,x in enumerate(self.weights):
            np.save(path+f"\\weights{k}", x)
        for k,x in enumerate(self.biases):
            np.save(path+f"\\biases{k}", x)

    def LoadDataVar(self, b, w):
        self.weights = w
        self.biases = b

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def CalculateError(result, desired): 
    e = np.array(desired) - np.array(result)    
    return e

def FilterData(Weight, Biase, sizes):
    Weights = []

    temp = []
    x = 0
    while x  <   sizes[0] * sizes[1]:
        temp.append(Weight[x])
        x+=1
    Weights.append(np.array(temp).reshape(sizes[1], sizes[0]))

    temp = []
    while x  <  (sizes[0] * sizes[1]) + (sizes[1] * sizes[2]):
        temp.append(Weight[x])
        x+=1
    Weights.append(np.array(temp).reshape(sizes[2], sizes[1]))

    temp = []
    while x  <  (sizes[0] * sizes[1]) + (sizes[1] * sizes[2]) + (sizes[2] * sizes[3]):
        temp.append(Weight[x])
        x+=1
    Weights.append(np.array(temp).reshape(sizes[3], sizes[2]))



    Biases = []

    temp = []
    x = 0
    while x  <  sizes[1]:
        temp.append(Biase[x])
        x+=1
    Biases.append(np.array(temp).reshape(sizes[1], 1))

    temp = []
    while x  <  sizes[1] + sizes[2]:
        temp.append(Biase[x])
        x+=1
    Biases.append(np.array(temp).reshape(sizes[2], 1))

    temp = []
    while x  <  sizes[1] + sizes[2] + sizes[3]:
        temp.append(Biase[x])
        x+=1
    Biases.append(np.array(temp).reshape(sizes[3], 1))

    return (Weights, Biases)

def ReturnNeuronIndex(Inputs):
    return [
             0 if np.array_equal(x, np.array([1, 0, 0, 1]).reshape(4,1).astype(np.single))
        else 1 if np.array_equal(x, np.array([1, 0, 1, 0]).reshape(4,1).astype(np.single))
        else 2 if np.array_equal(x, np.array([0, 0, 1, 1]).reshape(4,1).astype(np.single))
        else 3 if np.array_equal(x, np.array([0, 1, 1, 0]).reshape(4,1).astype(np.single))
        else 1 if np.array_equal(x, np.array([0, 1, 0, 1]).reshape(4,1).astype(np.single))
        else 2 if np.array_equal(x, np.array([1, 1, 0, 0]).reshape(4,1).astype(np.single))
        else -1
        for x in Inputs
    ]
#endregion

def PropCPU():
    Weight = np.full((20,1), .5).astype(np.single)
    Biase  = np.full((8, 1), .5).astype(np.single)
    Data = FilterData(Weight, Biase, [4, 2, 2, 4])
    Weight = Data[0]
    Biase  = Data[1]

    net = Network([4, 2, 2, 4])
    net.LoadDataVar(Biase, Weight)

    Inputs = [
        np.array([1, 0, 0, 1]).reshape(4,1).astype(np.single),
        np.array([1, 0, 1, 0]).reshape(4,1).astype(np.single),
        np.array([0, 0, 1, 1]).reshape(4,1).astype(np.single),
        np.array([0, 1, 1, 0]).reshape(4,1).astype(np.single),
        np.array([0, 1, 0, 1]).reshape(4,1).astype(np.single),
        np.array([1, 1, 0, 0]).reshape(4,1).astype(np.single)
    ]
    random.shuffle(Inputs)

    DI = ReturnNeuronIndex(Inputs)
    DN = np.diag(np.full(4,1))

    start=time.time()
    y = 0
    s = time.time()
    while y<500:
        for k,I in enumerate(Inputs):
            x=0
            while x<500:
                net.Backpropagate(I, DN[DI[k]].reshape(4,1), 1)
                x+=1
        y+=1
        if(y%100 == 0):
            print(f"Time took on batch {y/100}: {str(time.time()-s)}")
            s = time.time()
    print("Overall Time took: " + str(time.time()-start))

    print(net.evaluate(np.array([1, 0, 0, 1]).reshape(4,1).astype(np.single)))
    net.SaveData("TestingDataCPU")

def PropGPU(program, context, queue):
    sizes  = np.array([4, 2, 2, 4]).reshape(4,).astype(np.int32)

    Inputs = np.array( [
            1, 0, 0, 1, # \
            1, 0, 1, 0, # |
            0, 0, 1, 1, # _
            0, 1, 1, 0, # /
            0, 1, 0, 1, # |
            1, 1, 0, 0  # -
    ] ).reshape(24,1).astype(np.float64)

    DesiredN_Index = np.array( [ 0, 1, 2, 3, 1, 2 ] ).reshape(6,1).astype(np.int32)

    DesiredNNeurons = np.diag(np.full(sizes[3],1)).astype(np.int32)

    Weight = np.full((20,1), .5).astype(np.float64)
    Biase  = np.full((8, 1), .5).astype(np.float64)

    Data = FilterData(Weight, Biase, sizes)

    debug  = np.empty((2, 4)).astype(np.float64)

    buffer_I  = cl.Buffer(context, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR, hostbuf=Inputs)
    buffer_DN = cl.Buffer(context, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR, hostbuf=DesiredNNeurons)
    buffer_DO = cl.Buffer(context, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR, hostbuf=DesiredN_Index)
    buffer_W  = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=Weight)
    buffer_B  = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=Biase)
    buffer_S  = cl.Buffer(context, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR, hostbuf=sizes)
    buffer_D  = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, debug.nbytes)
    start=time.time()
    program.BackProp3Layer(queue, (1,1), None, buffer_I, np.float64(1), np.int32(6), np.int32(500), np.int32(500), buffer_DN, buffer_DO, buffer_W, buffer_B, buffer_S, buffer_D)

    net = Network([4, 2, 2, 4])

    cl.enqueue_copy(queue, Weight, buffer_W).wait()
    cl.enqueue_copy(queue, Biase,  buffer_B).wait()
    cl.enqueue_copy(queue, debug,  buffer_D).wait()
    
    os.system("cls")

    Data = FilterData(Weight, Biase, sizes)
    Weights = Data[0]
    Biases = Data[1]
    net.LoadDataVar(Biases, Weights)

    print("Time took: " + str(time.time() - start))
    print(net.evaluate(np.array([1,0,0,1]).reshape(4,1)))
    net.SaveData("TestingData")

os.system("cls")

# platform = cl.get_platforms()[0]
# device = platform.get_devices()[0]
# context = cl.Context([device])
# #context = cl.create_some_context()
# queue = cl.CommandQueue(context)

# kernel_code  = open("Kernel.c", "r").read()

# kernel_code  = kernel_code.replace("###SIZE:Layer0###", "4")
# kernel_code  = kernel_code.replace("###SIZE:Layer1###", "2")
# kernel_code  = kernel_code.replace("###SIZE:Layer2###", "2")
# kernel_code  = kernel_code.replace("###SIZE:Layer3###", "4")
# kernel_code  = kernel_code.replace("###SIZE:Layer1*Layer1###", "4")
# kernel_code  = kernel_code.replace("###SIZE:Layer2*Layer2###", "4")
# kernel_code  = kernel_code.replace("###SIZE:Layer3*Layer3###", "16")
# kernel_code  = kernel_code.replace("###SIZE:Layer2*Layer3###", "8")
# kernel_code  = kernel_code.replace("###SIZE:Layer1*Layer2###", "4")
# kernel_code  = kernel_code.replace("###SIZE:Weights0###", "8")
# kernel_code  = kernel_code.replace("###SIZE:Weights1###", "4")
# kernel_code  = kernel_code.replace("###SIZE:Weights2###", "8")
# kernel_code  = kernel_code.replace("###SIZE:Layers###", "4")

# program = cl.Program(context, kernel_code).build()
# PropGPU(program, context, queue)

PropCPU()