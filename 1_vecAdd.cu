#include<algorithm>
#include<cassert>
#include<iostream>
#include<vector>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU

// __restrict is a hint to the compiler that these pointers will not alias
// This can help the compiler optimize memory access
// __global__ functions can only return void, so we use pointers to write results  to the output vector

//  *_restict --> This pointer is the only one that will access the memory it points to.
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N) {
   // Calcuate global thread ID
   int tid = (blockIdx.x * blockDim.x) + threadIdx.x;


   // This prevents out-of-bounds memory access N is the size of the vectors
   // blockIdx.x is the block index in the grid, blockDim.x is the number of threads in each block, and threadIdx.x is the thread index within the block
   // Boundary check - To check if the thread ID is within the bounds of the vector size
   // If tid is less than N, we can safely access the vectors
   // This is a common pattern in CUDA to ensure that we do not access memory outside the allocated range
   // This is important for avoiding segmentation faults or undefined behavior
   if (tid < N) {
        // Add vecors parallelly 
        c[tid] = a[tid] + b[tid];

    //  Important Note: tid is the thread ID, which is a unique identifier for each thread in the grid not just within the thread block
    // Consider `tid` as the variable `i` 
   }
}

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c){
    for (int i=0; i < a.size(); i++){
        assert(c[i] == a[i] + b[i]);
    }
}

//  assert() --> The assert() function is a debugging aid that checks a condition at runtime. If the condition is false, the program:
// Prints an error message (with the failed condition and file/line number), Aborts execution (by calling abort()).
// To verify assumptions that should always be true during program execution. If they’re not, it indicates a logic error (bug) in your code.

// Think of assert() as a safety net to, Catch errors early during development. Help with unit testing or sanity checks.
// Avoid undefined behavior from bad data or logic bugs.




int main() {
    // Array size of 2^16 (65536 elements)
    constexpr int N = 1<<16;
    constexpr size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data 
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);


    // Initialize random numbers in each array 
    for (int i =0; i < N; i++){
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // CTA - Cooperative Thread Array.
    // Threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    //  We need to launch at LEAST as many threads as we have elements
    //  This pads an extra CTA to the grid if N cannot evenly be divided
    //  by NUM_THREADS (e.g. N =1025, NUM_THREADS = 1024)

    int NUM_BLOCKS = (N + NUM_THREADS -1) / NUM_THREADS;


    // launch the kernel on the GPU
    //  Kernle calls are asynchronus (the CPU Program continues execution after call, but no necessarily before the kernel finishes)
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    // Copy sum vector from device to host 
    // cudaMemcpy is a synchronus operation, and waits for the prior kernel launch to complete (both go to the 
    // default stream in this case).
    //  Therefore, this cudaMemcpy acts as both a memcpy and synchonization barrier.

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    verify_result(a,b,c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;

}