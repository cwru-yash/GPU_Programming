Goals:
    Understand Matrix Multiplication
    See how basic MM translates to GPU Execution
    Look at performance considerations for later optimizations.

Example:
    Multiplication of 2 square matrics with an identically sized square matrix as a result.
        
        A[(1,1),(1,2),           B[(1,1),(1,2),                       C[(1,1),(1,2),          
            (2,1),(2,2)           (2,1),(2,2)                         (2,1),(2,2)  
            ]                       ]                                   ]                          
                    
Basic Flow:
    1. Assign a thread dor ach element of C.
    2. Each thread traverses 1 row of A, and one column of B.
    3. Each thread wirtes the results to it's assigned element of C.

          C[(1,1)] = A[(1,1)] * B[(1,1)] + A[(1,2)] * B[(2,1)] --> Thread 1
          C[(1,2)] = A[(1,1)] * B[(1,2)] + A[(1,2)] * B[(2,2)] --> Thread 2
          C[(2,1)] = A[(2,1)] * B[(1,1)] + A[(2,2)] * B[(2,1)] --> Thread 3      
          C[(2,2)] = A[(2,1)] * B[(1,2)] + A[(2,2)] * B[(2,2)] --> Thread 4

2-D indexing for Thread Blocks:
    Tiny 2*2 thread blocks: 
    Variables availible:
        blockidx and threadidx:
            X and Y dimensions
        blockDim is constant:
            X and Y dimensions

To compute each element of c_ij of the result matrix C, We use:
    c_ij = a_i1 * b_1j + a_i2 * b_2j

CUDA Thread Mapping:
    If you launch a 2D grid of threads with dimensions (2,2), each thread is responsible for computing one element in the result matrix. The 2D thread indexing is done using:

        threadIdx.x (column index) --> int row = threadIdx.y; --> Rows varying vertically ⇒ threadIdx.y

        threadIdx.y (row index) --> int column = threadIdx.x; --> Columns varying horizontally ⇒ threadIdx.x

    NOTE: Most matrices in C/C++ are stored in row-major order, where the memory is laid out row by row. So, it's intuitive to think of the above.


Questions:
    1. Variable values for element [2,2]
        1. BlockIdx.x =1, BlockIdx.y =1
        2. ThreadIdx.x = 0, ThreadIdx.y =0
    2. Key Ideas ?
        1. Row = blockIdx.y * blockDim.y
        2. Col = blockIdx.x * blockDim.y


Total Global Thread Index Calculation:
To compute the global coordinates (row, col) of a thread in the grid:
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

Blocks are laid out as follows:

          blockIdx.x
         0       1
       +--------+--------+
block  |        |        |
Idx.y  |  (0,0) |  (1,0) |   ← blockIdx.y = 0 (bottom row)
       |        |        |
       +--------+--------+
       |        |        |
       |  (0,1) |  (1,1) |   ← blockIdx.y = 1 (top row)
       |        |        |
       +--------+--------+
Each labeled section is a block.

Block (1,1) is the top-right.

Block (0,0) is the bottom-left.

🟨 THREADS (threadIdx.x, threadIdx.y)
Inside each block, threads are arranged like this:

       threadIdx.x →
       0     1
     +-----+-----+
  0  |(0,0)|(0,1)|   ← threadIdx.y = 0 (top row)
     +-----+-----+
  1  |(1,0)|(1,1)|   ← threadIdx.y = 1 (bottom row)
These are local thread coordinates within a block.

threadIdx.x moves across columns

threadIdx.y moves down rows

