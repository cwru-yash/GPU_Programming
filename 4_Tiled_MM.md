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


1. Partitioning Matrices: Matrices A and B are divided into tiles of size TILE_SIZE x TILE_SIZE.
2. Loading into Shared Memory: Each thread block loads a tile of A and a tile of B into shared memory.
3. Computing Partial Results: Threads compute partial results using the tiles in shared memory.
4. Accumulating Results: Partial results are accumulated to compute the final value for each element in the result matrix C.
5. Storing Results: The final results are written back to global memory.

Benefits of Tiling:
Reduced Global Memory Accesses: By reusing data in shared memory, the number of global memory accesses is significantly reduced.

Improved Memory Coalescing: Access patterns can be optimized to ensure that threads access contiguous memory locations, enhancing memory throughput.

Enhanced Parallelism: Tiling allows for better utilization of the GPU's parallel processing capabilities by enabling concurrent computation of multiple tiles.


