Learn how programs execute in a GPU. 

Goal:
    Learn how programs execute ina GPU
    Learn the different granularities of threads in GPUs.
    Go through the most basic example of Vector Addition in CUDA.



The SIMT Model:
    Classical Execution Model:
        Sequential Vector Addition:
                A[0] + B[0] = C[0]
                A[1] + B[1] = C[1]
                A[2] + B[2] = C[2]
                A[3] + B[3] = C[3]
            We basically write a for loop and and do the individual addition operation with the given vectors (4 times in our case).

    Single Instruction Multiple Threads(SIMT):
        Parllel Vecotr Addition:
                A[0]   B[0]   C[0]
                A[1] + B[1] = C[1]
                A[2]   B[2]   C[2]
                A[3]   B[3]   C[3]
            A single instruction will be executed on multiple trheads which in the above case says that we want to do a vector based operation (addition) and the result will also be an array which is a natural way of mapping parallel data in a very clean way. 


From Threads to Grids:

    Threads:
        Lowest granualrity of execution
        Executes instructions

    Warps (SIMT!): Threads are collectively wrapped as warps
        Lowest schedulable entity
        Executes instructions in lock-step                          
            Not every thread needs to execute all instructions --> Depending on the size of the problem, some of these threads will be masked off and they're not executing the same instruction or for that matter any instruction at the same time, but in general they're all executing the same instruction at a given time.

    Thread Blocks: Warps together are composed in thread blocks or `blocks` more commonly referred as.
        Lowest Programmable entity
        Is assigned to a single shader core
        Can be 3-D --> you can have thread blocks in the x direction, y direction and the z direction, this helps in mapping the problem to the actual threads themselves.
    
    Grids: Thread blocks are composed into Grids
        How a problem is mapped to the GPU --> this is where it's determined
        Part of the GPU launch parameters --> Launch parameters are grid size and how many parameters we've in each dimension and how many threads are there within each thread block 
        Can also be 3-D --> 

Baseline GPU Architecture:

Shader Cores:
    Witihn each SM we've our Warp schedulers that actually starts execution inside of each of these cores.
    Then within these cores we've things like texture caches and l1 Caches, Shared Memory (sometimes called as scratchpad memory)  and we've our ALUs,  Double Precision (DP) Units, Special Functions (SF) Unit, Load Storage Units (LD/ST),e tc. 


🔍 What is PTX and Why It’s Used?
PTX (Parallel Thread Execution) is NVIDIA’s virtual instruction set architecture (ISA) for CUDA. Think of it as an intermediate language between CUDA C++ code and the actual machine code executed by the GPU. Here’s why it matters:

Portability: PTX allows your code to be compiled for different GPU architectures without rewriting it.

Optimization: NVCC can generate PTX which is then optimized by the driver at runtime for the specific GPU.

Debugging & Analysis: Viewing PTX gives insights into what the GPU is really executing.

🧠 How to Inspect PTX for Your Code
Compile to PTX:


nvcc -ptx your_code.cu -o vectorAdd.ptx
Read the PTX file:

less vectorAdd.ptx