Floating-point rounding error
When you multiply and add floats on the GPU (or on the CPU), each operation generally rounds its result to the nearest representable 32-bit value. Even if two different sequences of multiply-add operations compute mathematically the same result, they can round in slightly different ways. Over many accumulations (as in an 𝑛-term dot product) these tiny per-operation errors can add up to a non-zero difference.

Why not test for exact equality?
If you wrote


assert(c[j * n + i] == temp);
it would very often fail, because the computed c[j * n + i] and your host’s temp could differ by maybe one or two ULPs (units in the last place) even though both are perfectly reasonable floating-point results of the same mathematical dot product.

Using an epsilon tolerance

Instead, we pick a small threshold epsilon = 0.001f; and check


fabs(c[...] - temp) < epsilon
This means “the two results are within 0.001 of each other.” That tolerance should exceed the worst-case accumulation of rounding error over 𝑛 products and adds, yet be small enough that any bug (e.g. a wrong index or uninitialized value) would almost certainly produce a difference larger than 0.001.


Leading Dimensions:

    In cuBLAS (and all BLAS libraries), the **leading dimension** parameters (`lda`, `ldb`, `ldc`) tell the library how the matrices are laid out *in memory*, specifically the “stride” you must take when you move from one column to the next (in column-major order) or one row to the next (in row-major order).

---

### 1. What is the leading dimension?

The leading dimension of a matrix is simply the length of its first dimension *as stored in memory*.  In a column-major layout (the default for Fortran and cuBLAS), that means

```
lda = (number of rows allocated in memory for A)  
```

regardless of whether you’re using the full matrix or just a sub-matrix. ([stackoverflow.com][1])

---

### 2. Why “first dimension”?

BLAS routines assume **column-major** storage: elements of column \$j`are contiguous, and to go from A(0,j) to A(0,j+1) (elements we need to skip over an entire column vertically to reach the next column) you must skip over all the rows of column`j`.  That skip—how many floats (or doubles) you jump over—is exactly `lda\`.  In math terms, the element  $A_{i,j}$ lives at memory address

```
& A[  j*lda  +  i  ]
```

where `i` runs from `0` to `m−1`. ([docs.nvidia.com][2])

---

### 3. Why do we use it?

1. **Sub-matrix and padding support.**  If you allocate a large array but only use an $m\times n$ corner of it, you still need to tell BLAS how wide the *allocated* storage is.  By passing `lda`, you can work on sub-matrices without copying them into a perfectly packed buffer.
2. **Flexible strides.**  In more advanced scenarios (e.g. batched GEMMs, strided data, pitch-linear allocations) your matrix might not be “tightly packed.”  `lda` lets you express arbitrary row-or-column strides.
3. **Performance tuning.**  Sometimes padding rows to a multiple of 32 or 64 can improve coalescing or cache behavior; you simply set `lda` to the padded row count and BLAS will still step correctly through memory. ([salykova.github.io][3])

---

#### In your call:

```cpp
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha,
            d_a, /* lda = */ n,
            d_b, /* ldb = */ n,
            &beta,
            d_c, /* ldc = */ n);
```

Here each matrix is $n\times n$ and *tightly packed*, so you set `lda = ldb = ldc = n`.  If you ever stored your data with extra padding (say 2 floats per row), you’d still call with `lda = n+2`, and cuBLAS would correctly skip the padding when moving column-to-column.

