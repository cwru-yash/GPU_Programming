Goals:
    Understand another key aspect of GPU code performance.
    Show how memory alingment impacts code performance.



Recall Row-major Order: 
    Programming languages can gives us a convenient abstraction of memory
        That doesn't mean memory is built that way!
        Recall what a matrix looks like,
            To the Programmer.
            In terms of memory addresses


For Aligned Accessses:
    Consider the B-Matrix access pattern in matrix multiplication
        Each thread accesses a different column
        Columns are adjacent in memory
            Mulitple adjacent accesses can be coalesced into a single wide access.


For misaligned Accesses:
    Consider the A-Mtrix access pattern in matrix multiplication
        Each thread access a different row
        Rows are NOT adjacent in memory
            Multiple accesses to memory that are independent
            Rows can be very long!


to address this problem of misaligned access 
