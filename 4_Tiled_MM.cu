//  This program conputes matrix multiplication using shared memory tiling

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

//  Pull out matrix and shared memory tile size
// const int N = 1<<10;
// const int SHEMM_SIZE =1<<10;

//  To run on  RTX 3070
const int N = 1<<5;
const int SHEMM_SIZE =1<<5;

__global__ void matrixMul(const int *a, const int *b, int *c){
    // Cpompute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("row %d", &row);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("column %d", &col);

    if (row < 2 && col < 2) {
        printf("row %d, col %d\n", row, col);
    }
    

    // Statically allocated shared memory --> We allocate it statically
    // __shared__ int s_a[SHEMM_SIZE];  // This is how we tell our device to use shared memory for this code
    // __shared__ int s_b[SHEMM_SIZE];
    
    //  Minor Fix to be done with my code to reduce the size of the matrix, such that My code works on RTX 3070.
    __shared__ int s_a[SHEMM_SIZE * SHEMM_SIZE];
    __shared__ int s_b[SHEMM_SIZE * SHEMM_SIZE];
    


    // Accumulate in temporary variable
    int tmp = 0;

    //  Sweep tile across matrix --> Here, N = n/ tile_size -- total size of the n*n-matirx/tile_size
    for (int i =0; i< N; i+= blockDim.x){
        /* 
        Every thread in a threadblock loads one element into shared memory. The element into shared memory
        the element location ins shared memory corresponds to the thread's position in the threadblock (e.g. thread [0,0] loads for 
        A[0* tile_size + 0], and B[0 * tile_size + 0].)

        Explanation of indexig parameters

        For A;
                row*n:          Indexes the global row for this thread (loop-invariant)--> that we're going to work on
                i*tile_size:    Indexes the new set of columns of each iteration --> the tile only has a subset of columns 
                                that we're working on and this gives us that subset of columns that we're currently working on.
                tx:             Indexes the column within that set
                                
         
        For B:
                i*tile_size*n:  Indexes the next set of rows each iteration --> for each iteration we've a small subset of rows 
                                that we're going to be working on and this gives us that subset of rows that we're currnly working on, 
                                also if we want to switch to the next row within the memory we will have to think about how it si indexed within memory
                                or laid out in memory, so to move to the next spot in memory we've to go over to the entire block that would be associated with the entire row
                                so we've to multiply by the matrix size.
                ty*n:           Indexes the row within that set. --> to get a specific row within the set of rows  we do or threadidx.y (up and down) * matrix size (n) 
                                No. of elements between th2 rows will be n elements hence we use this.           
                col:            Indexes the global column (loop-invariant) --> that w're going to work on
        */ 

        // Load in elements for this tile--> into shared memory using manual 2D-to-1D index Mapping
        s_a[threadIdx.y *blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x]; // --> every single thread loads a single element from A to shared memory of the thread Block A
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        // Wait for both tiles to be loaded in before doing computation
        __syncthreads(); // --> every single thread within thsi block needs to be done by at this point.
        //  We do this to get all our threads in sync before the loop iteration such taht we don't stomp our memory and
        //  run into segmentation faults and indexing errors. We load it with the assumption that even though only one thread is going 
        //  to load a value in the shared memory, many threads are going to access it, hence we need to call __syncthreads() such taht 
        //  we can have all our thread before they proceed.


        // Do matrix multiplication on the small matrix
        for (int j =0; j < blockDim.x; j++){ // Every thread sweep across the rows within the tile and the column within the tiles.
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
            //  For matrix A --> this will be the row that we're in and this will only be changed by the column we're in.
            //  Similarly, this will be the opposite of B, the row we're going to be accessing since we're going downward
            //  will be changing with a factor of j*tile_size and then a loop invariant column and then accessing which column we're going to be on.
            // within that tile.
        }

        //  Wait for all threads to finish using current tiles before loading new ones
        __syncthreads(); //--> so some of the values still haven't been writted to the shared memory and we write them
        //  nd access them later --> this __syncthread() actually helps us in syncing when people have has stopped accessing the shared memory.
        
        }
        
        // Write back results
        c[row * N + col] = tmp; // --> now we write to DRAM now the tmp value that we've been storing and acuumulating in the loop above.

    }

    // Check result on the CPU
    void verify_result(vector<int> &a, vector<int> &b, vector<int> &c){
        // For every row... // (in A)
        for(int i = 0; i <N; i++){
        //  For every column... // (in B)
        for (int j=0; j< N; j++){
            //  For every elemnt in the row-column pair 
            int tmp =0;
            for (int k=0; k<N; k++){
                //  Accumulate the partial results
                tmp +=a[i*N + k] *b[k * N + j];
            }

            //  Check against the CPU result
            assert(tmp == c[i * N + j]);
        }
        
        }
        }
    int main(){
        // Size (in bytes) of matrix --> Good technique (remember this)
        size_t bytes = N * N * sizeof(int);

        //  Host vectors
        vector<int> h_a(N*N);
        vector<int> h_b(N*N);
        vector<int> h_c(N*N);

        // Initialize matices
        generate(h_a.begin(), h_a.end(), []() {return rand()%100; });
        generate(h_b.begin(), h_b.end(), []() {return rand()%100; });

        //  Allocate decvice memory
        int *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        // Copy data to the device 
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

        //  Threads per CTA dimension
        int THREADS = 32;

        //  Blocks per grid dimension (assumes THREADS divides N evenly)

        int BLOCKS = N / THREADS;

        //  Use dim3 structs for block and grid dimensions
        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS, BLOCKS);

        //  launch kernel
        matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

        //  Copy  back to the host 
        cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

        // Check result
        verify_result(h_a, h_b, h_c);

        cout << "COMPLETED SUCCESSFULLY\n";

        //  Free memory on device
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return 0;
    }
    