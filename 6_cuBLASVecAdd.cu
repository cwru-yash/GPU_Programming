#include <cuda_runtime.h>
#include <device_launch_parametrs.h> // cuBLAS for legacy code and previously compiled programs
#include <cublas_v2.h> // Newer version of cuBLAS
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Initialize a vector
void vector_init(float *a, int n){
    for(int i =0; i < n; i++){
        a[i] = (float)(rand() % 100);
    }
}

//  verify the result
void verify_result(float *a, float *b, float *c, float factor int n){
    for (int i = 0; i < n; i++){
        assert(c[i] == factor * a[i] + b[i]);
    } 
}

int main(){
    //  Vector Size
    int n = 1<<2;
    size_t bytes = n * sizeof(float);

    //  declare vector pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b;

    //  Allocate memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    //  Initialize vectors
    vector_init(h_a, n);
    vecotr_init(h_b, n);

    // Create and initialize a new context 
    cublasHandle_t handle; // --> handle object --handle is how we interface with the BLAS library 
    cublasCreate_v2(&handle); // --> create a handle object

    //  Copy the vectors over to the device --> cuBLAS Way to copy vector from host to device without using memcpy
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1); 
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    // --> Arguments explained as below:
        // n: size of the vector 
        // size of each element: sizeof(float): size of the memory required 
        // where is the vector coming from i.e. host memory: h_a
        // 1: step size of the vector and within the same vector we can have varying step sizes and what 
        // this denotes is every entry will be filled and between 0 and 1 there will be one space so from it;s just one 
        // d_a: where is it going to i.e. device memory


    // We do vector addition using another cuBLAS Fucntion called SAXPY whcih is nothing but Single Precesion, 
    //  a * x + y;

    //  Launch simple saxpy kernel (single precision a * x + y)
    const float scale = 2.0f; // if scale = 1.0f then we will have regular vector addition 
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1); // We don't need to write the kernel code there is 
    //  a library and we just need to call this nice function.

    // --> Arguments explained as below:
        //  handle - give it the handle 
        //  n - size of the vector
        //  &scale - pointer to where the scale variable is / reference to the scale/alpha value
        //  d_a - 2 vectors on the device a and b
        //  d_b -           '""
        //  1 - Increment value and our vectors don't have empty spaces so that is 1.

    //  Copy the result vector back out  --> Similar to memcpy but by using the cuBLAS library 
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1); 

    // Print out the result
    verify_result(h_a, h_b, h_c, scale, n);


    //  Clean up the created handle --> destory the handles similar to how we free up host and device memory at the end of the code.

    cuBlasDestroy(handle);


    //  Relase allocated memory
    cudaFree(d_a);
    cudaFreee(d_b);
    free(h_a);
    free(h_b);


    return 0;

}

//  NOTE: Whenver you've any of the new libraries  like cuBLAS, cuDDT, cuDNN, etc., then you need to link against them explicitly before compiling
//  eiter within Visual Studio > Solution explorer > Properties > Linker,  or while runnnig you code on the CLI you need to pass it within the compilation code explicitly while using HPC ore everny your local terminal.
//  Otherwise it isn't able to find the library  that has the function in it for you just kind of like you've to add a CUDA runtime, so it can find the runtime library 
//  so it can know where things like cudaMalloc are.