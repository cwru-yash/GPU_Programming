Goals:
    Understand why cache tiling is a useful optimization
    Show how tiling can be applied to Matrix multiplication.


Cache Tiling:
    DRAM is slow: --> We were memcpy the 2 input matrices within our DRAM and then we had to load them in while doing out calculation while doing a GEMM. We want to leverage our L1 and L2 caches and avoid using main memory all the time since it's pretty slow. If we've fewer memory related stalls then fewer of our time is psent towards computation or doing useful work.
        It would be nice if we had a way to gaurantee something is in the cache
        Fewer memory related stalls, more computation/unit time

    Solution?:  
        Shared Memory(Scratchpad):
            User-managed L1 Cache --> The programmer load values inside the shared memroy and it stays there, in this way we get a 5-10 cycles time in shared memory otherwise it would take to go all the way out to DRAM.
            Private per-threadblock --> 

        Another one of the reasons of using cache-tiling is that the pieces of the large input that we're using right now in the cache and whatevers in the cache is useful so that;s what we do with the sahred memory.

Tiled Matrix Multiplication:
    Within our Naive or GEMM we had a single thread that calculate a single element of the Matrix C, i.e c_ij = a_ik * b_kj --> computed by a single thread.
    Now each thread instead of sweeping across all of A, it sweeps across a subset of this matrix i.e. our tile.

    NOTE: The matrix size and the tile size must be evenly or fully divisible ideally an exponenet of 2. 

    Calculating index for loading into shared memory:
        Constant row, loop varying column
        Constant column, loop varying row

    Actual calulation?
        A[y][k]* B[k][x]
            Row is loop invariant for A --> Constant row matrix threads here progress along and the row doesn't changes only the column changes for each tile.
            Column is loop invariant for B. --> Similarly for the column matrix the threads here progress along and the column doesn't changes only the row changes.
        In case of our Naive MM we had to figure out what global row or column value of c or of A and B are we grabbing from? 
         But in the case of out Tile cached MM we're grabbing everything into the tiles, we just index purely by out threadid within that actual tile. Tiles in this case are our proxy fo Thread Blocks, so our TILE_SIZE will be similar (or same in most cases) to our Thread-Block size.

         We're taking a large input and reducing it to the size of our thread blocks at each loop iteration
         



🔧 1. Computing Partial Results
⏬ Step-by-Step:
Each thread block loads a tile of matrix A and matrix B into shared memory.

Suppose the tile size is TILE_SIZE x TILE_SIZE.

Each thread in the block computes one element of the result submatrix Ctile using:

            Ctile[i][j]+=Ashared[i][k]∗Bshared[k][j]
This is just one tile-tile multiplication step.

🔄 Example:
For a tile from A and B:

Ashared = [[1, 2],
           [5, 6]]

Bshared = [[16, 15],
           [12, 11]]
Then for thread (0, 0) in the tile, it computes:

C[0][0]+=1×16+2×12=16+24=40
Every thread in the tile does this for its corresponding (i, j) element.

🔄 2. Accumulating Results
Since a full matrix multiplication involves summing over the shared axis, say for matrix dimensions:
                            Ci,j=k∑Ai,k⋅Bk,j
​
 
You loop over all tile positions along the shared dimension (commonly k) and keep adding the partial results computed from the tiles. That’s what “accumulation” means.

Each thread holds a running total like:

    float temp = 0;
    for (int t = 0; t < numTiles; ++t) {
        temp += Ashared[threadIdx.y][t] * Bshared[t][threadIdx.x];
    }
    C[row * N + col] = temp;


1. Partitioning Matrices: Matrices A and B are divided into tiles of size TILE_SIZE x TILE_SIZE.
2. Loading into Shared Memory: Each thread block loads a tile of A and a tile of B into shared memory.
3. Computing Partial Results: Threads compute partial results using the tiles in shared memory.
4. Accumulating Results: Partial results are accumulated to compute the final value for each element in the result matrix C.
5. Storing Results: The final results are written back to global memory.

Benefits of Tiling:
Reduced Global Memory Accesses: By reusing data in shared memory, the number of global memory accesses is significantly reduced.

Improved Memory Coalescing: Access patterns can be optimized to ensure that threads access contiguous memory locations, enhancing memory throughput.

Enhanced Parallelism: Tiling allows for better utilization of the GPU's parallel processing capabilities by enabling concurrent computation of multiple tiles.


