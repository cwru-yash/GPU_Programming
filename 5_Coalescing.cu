#include <stdlib.h>
#include <math.h>
#include <assert.h>


__global__ void matrixMul(int *a, int *b, int *c, int n) {
    //  Compute each thread's row
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Compute each thread's column
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum =0;
    //  Boundary protection
    if ((row < n) && (col < n)) {
        //  Iterate over row, and down column
        for (int k =0; k <n; k++){
            // Accumulate result for a single element
            temp_sum += a[k * n + row] * b[k * n  + col];
        }
        //  Assign result
        c[row * n + col] = temp_sum;
    }

}

//  Initialization fucntion for matrices
void matrix_init(int *a, int n){
    for (int i =0; i < n; i++){
        for (int j =0; j < n; j++){
            a[i * n + j] = rand() % 100;
        }
    }
}

//  Check result
void check_answer(int *a, int *b, int *c, int n){
    int *verify_c;
    verify_c = (int*)malloc(n*n * sizeof(int));
    int temp_sum;
    for (int i =0; i <n; i++){
        for(int j=0; j <n; j++){
            temp_sum =0;
            for(int k=0; k<n; k++){
                temp_sum +=a[i * n + k] *b[k * n + j];
            }
            verify_c[i * n + j] = temp_sum;
        }
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++ ){
            assert(c[i * n + j] ==verify_c[i *n + j]);
        }
    }
}


//  Transpose a matrix
void transpose(int *a, int *a_t, int n){
    for (int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            a_t[j * n +i] = a[ i * n + j];
        }
    }
}

int main() {
    //  matrix size of 1024 x 1024;
    int n =1 <<10;

    //  Size (in bytes) of matrix
    size_t bytes = n *n * sizeof(int);

}