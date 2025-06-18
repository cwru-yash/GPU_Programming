//  This program calculates matrix multiplication (SGEMM) using cuBLAS
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
// Included for all the cuBLAS functions
#include <curand.h>
// curand is a randomization library which we will use to initialize both our matrices on the GPU side and 
//  this library will initialize our kernels and help us do that on the GPU side.
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// IMPORTANT NOTE: In c/C++ we assume that memory is layed out as in a row-major order where we assume that the rows are contigous
//  in memory one after the other in a horzontal format and we offset through them to access a particular data value
// However for cuBLAS it follow a column-major order, and in this case it means that the entire column is contigous in 
// memory but as column (i.e. vertically) and henceforth we had to change our indexing code below


void verify_solution(float *a, float *b, float *c, int n) {
    float temp;
    float epsilon = 0.001; // first time we use floating point numbers.
    for (int i =0; i < n; i++){
        for(int j =0; j < n; j++){
            temp =0;
            for(int k=0; k < n; k++){
                temp += a[k * n + i] * b[j * n + k]; // We modify our indexing here , Instead of a[i * n + k] and b[k * n + j] we've the given.
            }
            assert(fabs(c[j * n + i]-temp)< epsilon);
            // We're using floating point numbers which are tricky since they can produce weird results when rounded off,
            //  Hence we compare with an epsilon value of the result since we don't care for the number to be an exact match
            //  but we want it to be close enough 

            //  now we take absolute value of our orignal and temporary value and and hence we've a value within
            //  a close range of our orignal value because our epsilon value is 0.001
        }
    }
}

int main(){
    //  Problem size
    int n << 10;
    size_t bytes = n * n * sizeof(float);

    //  Declare pointer to matrices on device and host 
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Pseudo random number generator
    curandGenerator_t prng; // Similar to our cuBLAS handle we need a cuBLAS curand object for our random number generator.
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT); // To the curandCreateGenerator function we give our curand object and the default curand random generator object. 

    //  Set the seed --> mainly if you want to reproduce your results (since same seed will produce the same randomization) you use the same seed in our case we use the clock() function to get the timestamp and use the current timestamp (in GMT in seconds (By Default)) and use that number as our seed value.
    curandSetPseudorandomGeneratorSeed(prng, (unsigned long long)clock());

    //  Fill the matrixwith random numbers on the device
    curandGenerateUniform(prng, d_a, n*n); // Generates our random numbers from a uniform distribution.
    curanfGenerateUniform(prng, d_b, n*n); // We give our curand handler, our device pointer and the number of elements.
    // NOTE: Here we don't have to copy our data to the GPU since we've already allocated memory over on the GPU and since we generated our random numbers as well 
    //  on the GPU, we don;t have the need to copy it.


    //  cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scaling factors -- isnce it's a generalized form of Matrix Multiplication it has these values of alpha and beta that 
    //  we can set.
    float alpha = 1.0f;
    float beta  = 0.0f; // We've set beta to 0 such that we only get the matrix multiplication i.e. c =a*b formt he below mentioned equation.

    // Calulate: c = (alpha*a)*b + (beta*c) // --> here a, b, c are all matrices.
    //  (m X n) * (n X k) = (m X k)

    //  Signature: handle, operationm operation, m, n,k, alpha, A, lda, B, ldb, beta, C, ldc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
    //  Menaing of the Given Parameters below:
    // handle --> our cuBLAS handle
    // CUBLAS_OP_N --> are for convienince things we give are matrix to this object and it will do the computations on it as is
    // CUBLAS_OP_N --> it will transpose the given matirix for us and then do all the operations on the transposed matric.
    // n,n,n --> are the values of  m, n, k as defined in the matrix multilication equation --> (m X n) * (n * k) = (m X k)
    // &alpha --> pointer to where our alpha is stored, pass by reference.
    // d_a, n, d_b, n, d_c, n --> device pointers of a, b, and c, respectively and n in all the cases are their respective leading 
    // dimensions, so n -> leading dimension of a (lda) a square matrix with dimesion `n`, n --> ldb, n --> ldc.
    // &beta --> value of beta (pass by reference).

    //  Copy back the three matrices
    cudaMemcoy(h_a, d_a, bytes, cudaMemcpyDeviceToHost); // We could've also called cublasgetmatrix() however it's optional 
    cudaMemcoy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcoy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // verify solution
    verify_solution(h-a, H_b, h_c, n);

    printf("COMPLETED SUCCESSFULLY\n");
}