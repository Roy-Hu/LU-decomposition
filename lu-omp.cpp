#include <iostream>
#include <numa.h>
#include <omp.h>
#include <vector>
#include <cmath> 
#include <random> // For random number generation
#include <ctime> // For seeding the random number generator

using namespace std;

int value = 0;

// Function to allocate memory for a matrix
double** allocateMatrix(int n) {
    double** matrix = new double*[n];
    for (int i = 0; i < n; ++i) {
        matrix[i] = new double[n];
    }
    return matrix;
}

// Function to deallocate memory for a matrix
void deallocateMatrix(double** matrix, int n) {
    for (int i = 0; i < n; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Function to compute the product of matrices L and U
double** multiplyLU(double** L, double** U, int n) {
    double** LU = allocateMatrix(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            LU[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                LU[i][j] += L[i][k] * U[k][j];
            }
        }
    }
    return LU;
}

// Function to compute the product of permutation vector pi and matrix A, resulting in PA
double** multiplyPA(double** A, int* pi, int n) {
    double** PA = allocateMatrix(n);
    for (int i = 0; i < n; ++i) {
        // Use pi to permute rows when copying from A to PA
        for (int j = 0; j < n; ++j) {
            PA[i][j] = A[pi[i]][j];
        }
    }
    return PA;
}

// Function to compute the L2,1 norm of a matrix
double computeL21Norm(int n, double** A, double** L, double** U, int* P) {
    double** PA = multiplyPA(A, P, n);
    double** LU = multiplyLU(L, U, n);

    // Compute the residual R = PA - LU
    double** R = allocateMatrix(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            R[i][j] = PA[i][j] - LU[i][j];
        }
    }

    // Compute the L2,1 norm of the residual
    double norm = 0;
    for (int j = 0; j < n; ++j) {
        double colNorm = 0;
        for (int i = 0; i < n; ++i) {
            colNorm += R[i][j] * R[i][j];
        }
        norm += sqrt(colNorm);
    }

    // Deallocate memory
    deallocateMatrix(PA, n);
    deallocateMatrix(LU, n);
    deallocateMatrix(R, n);

    return norm;
}


// Function to perform LU decomposition
void LU_Decomposition(double** A, int n, int* pi, double** L, double** U) {
    // Create a copy of A to avoid modifying the original matrix
    double **A_prime = allocateMatrix(n);
    
    // Initialize L, U, and pi
    for (int i = 0; i < n; ++i) {
        pi[i] = i;
        for (int j = 0; j < n; ++j) {
            L[i][j] = (i == j) ? 1 : 0;
            U[i][j] = 0;
            A_prime[i][j] = A[i][j];
        }
    }
    
    // Perform LU decomposition
    for (int k = 0; k < n; ++k) {
        double max = 0;
        int k_prime = -1;
        for (int i = k; i < n; ++i) {
            if (max < abs(A[i][k])) {
                max = abs(A[i][k]);
                k_prime = i;
            }
        }
        
        if (max == 0) {
            cerr << "Error: Singular matrix" << endl;
            exit(EXIT_FAILURE);
        }

        swap(pi[k], pi[k_prime]);
        swap(A_prime[k], A_prime[k_prime]);
        for (int i = 0; i < k; ++i) {
            swap(L[k][i], L[k_prime][i]);
        }
        
        U[k][k] = A_prime[k][k];

        for (int i = k + 1; i < n; ++i) {
            L[i][k] = A_prime[i][k] / U[k][k];
            U[k][i] = A_prime[k][i];
        }

        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A_prime[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    deallocateMatrix(A_prime, n);
}

long fib(int n)
{
  if (n < 2) return n;
  else return fib(n-1) + fib(n-2);
}

void 
usage(const char *name)
{
	cout << "usage: " << name
                  << " matrix-size nworkers"
                  << endl;
 	exit(-1);
}


int
main(int argc, char **argv)
{

    const char *name = argv[0];

    if (argc < 3) usage(name);

    int matrix_size = atoi(argv[1]);

    int nworkers = atoi(argv[2]);

    cout << name << ": " 
                << matrix_size << " " << nworkers
                << endl;

    omp_set_num_threads(nworkers);

    // Allocate memory for matrices A, L, and U, and permutation vector pi
    double** A = allocateMatrix(matrix_size);
    double** L = allocateMatrix(matrix_size);
    double** U = allocateMatrix(matrix_size);
    int* pi = new int[matrix_size];
    
    // Random number generation setup
    mt19937 generator(time(nullptr)); // Random number generator seeded with current time
    uniform_real_distribution<double> distribution(-10.0, 10.0); // Range of random values

    // Generate random matrix A
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            A[i][j] = distribution(generator); // Assign a random double within the range
        }
    }

    // Proceed with LU decomposition and computing the L2,1 norm
    LU_Decomposition(A, matrix_size, pi, L, U);

    // Compute and print the L2,1 norm of the residual
    double L21Norm = computeL21Norm(matrix_size, A, L, U, pi);
    cout << "L2,1 norm of the residual: " << L21Norm << endl;

    // Deallocate memory
    deallocateMatrix(A, matrix_size);
    deallocateMatrix(L, matrix_size);
    deallocateMatrix(U, matrix_size);

    delete[] pi;

// #pragma omp parallel
// {
//    int tid = omp_get_thread_num();
//    void *mem = numa_alloc_local(4);
//    int myN = 20 - tid;
//    if (myN < 16) myN = 16;
//    long res = fib(myN);
// // #pragma omp critical
//    value = tid; // data race
//    printf("thread %d fib(%d) = %ld\n", tid, myN, res);
// }

    return 0;
}
