#include <iostream>
#include <numa.h>
#include <omp.h>
#include <vector>
#include <cmath> 
#include <random> // For random number generation
#include <ctime> // For seeding the random number generator
#include <sys/time.h>

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

// Function to deallocate memory for a matrix
void deallocateMatrix_numa(double** matrix, int n) {
    for (int i = 0; i < n; ++i) {
        numa_free(matrix[i], sizeof(double) * n);
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
double computeL21Norm(int nworkers, int n, double** A, double** L, double** U, int* P) {
    double** PA = multiplyPA(A, P, n);
    double** LU = multiplyLU( L, U, n);

    // Compute the residual R = PA - LU
    double** R = allocateMatrix(n);

    omp_set_num_threads(16);

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            R[i][j] = PA[i][j] - LU[i][j];
        }
    }

    // Compute the L2,1 norm of the residual
    double norm = 0;
    #pragma omp parallel for reduction(+:norm)
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

struct MaxKPrime {
    double max;
    int k_prime;
};

#pragma omp declare reduction(maximum : struct MaxKPrime : omp_out = omp_in.max > omp_out.max ? omp_in : omp_out)

void LU_Decomposition(int nworkers, double** A, int n, int* pi, double** L, double** U) {
    double **A_prime = new double*[n];

    #pragma omp parallel for num_threads(nworkers) shared(n, L, U, A, pi, A_prime, nworkers) default(none)
    for (int i = 0; i < n; ++i) {
        int numa_node = i % numa_num_configured_nodes();
        L[i] = (double *)numa_alloc_onnode(n * sizeof(double), numa_node);
        U[i] = (double *)numa_alloc_onnode(n * sizeof(double), numa_node);
        A_prime[i] = (double *)numa_alloc_onnode(n * sizeof(double), numa_node);

        pi[i] = i;

        for (int j = 0; j < n; ++j) {
            L[i][j] = (i == j) ? 1 : 0;
            U[i][j] = 0;
            A_prime[i][j] = A[i][j];
        }
    }

    for (int k = 0; k < n; ++k) {
        MaxKPrime max_k_prime = {0.0, -1};

        #pragma omp parallel num_threads(nworkers) shared(k, n, A, nworkers, max_k_prime) default(none)
        {
            MaxKPrime max_k_prime_private = {0.0, -1};

            // reduction operation for max_k_prime
            #pragma omp for nowait
            for (int i = k; i < n; ++i) {
                double abs_val = abs(A[i][k]);
                if (abs_val > max_k_prime_private.max) {
                    max_k_prime_private.max = abs_val;
                    max_k_prime_private.k_prime = i;
                }
            }

            //maximum values from all threads are combined in a critical section to find the overall maximum
            #pragma omp critical
            {
                if (max_k_prime_private.max > max_k_prime.max) {
                    max_k_prime = max_k_prime_private;
                }
            }
        }

        swap(pi[k], pi[max_k_prime.k_prime]);
        swap(A_prime[k], A_prime[max_k_prime.k_prime]);

        U[k][k] = A_prime[k][k];

        // #pragma omp parallel num_threads(nworkers) shared(k, n, L, U, A_prime, nworkers, max_k_prime) default(none)
        // {
        //     int tid = omp_get_thread_num();

        //     int numa_node = (k + 1 + tid) % numa_num_configured_nodes();
        //     numa_run_on_node(numa_node);
        //     numa_set_localalloc();

        //     for (int i = tid; i < k; i += nworkers) {
        //         swap(L[k][i], L[max_k_prime.k_prime][i]);
        //     }

        //     #pragma omp barrier

        //     for (int i = k + 1 + tid; i < n; i += nworkers) {
        //         L[i][k] = A_prime[i][k] / U[k][k];
        //         U[k][i] = A_prime[k][i];
        //     }

        //     #pragma omp barrier

        //     for (int i = k + 1 + tid; i < n; i += nworkers) {
        //         for (int j = k + 1; j < n; ++j) {
        //             A_prime[i][j] -= L[i][k] * U[k][j];
        //         }
        //     }
        // }
 
        #pragma omp parallel for num_threads(nworkers)
        for (int i = 0; i < k; i++) {
            swap(L[k][i], L[max_k_prime.k_prime][i]);
        }

        U[k][k] = A_prime[k][k];

        #pragma omp parallel for num_threads(nworkers)
        for (int i = k + 1; i < n; ++i) {
            L[i][k] = A_prime[i][k] / U[k][k];
            U[k][i] = A_prime[k][i];
        }

        #pragma omp parallel for num_threads(nworkers)
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A_prime[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    deallocateMatrix_numa(A_prime, n);
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

    if (numa_available() == -1) {
        cerr << "Error: NUMA unavailable" << endl;
        exit(EXIT_FAILURE);
    }

    printf("Number of NUMA nodes: %d\n", numa_num_configured_nodes());
    // Allocate memory for matrices A, L, and U, and permutation vector pi
    double** A = allocateMatrix(matrix_size);
    double** L = new double*[matrix_size];
    double** U = new double*[matrix_size];

    int* pi = new int[matrix_size];
    
    // Random number generation setup
    mt19937 generator(time(nullptr)); // Random number generator seeded with current time
    uniform_real_distribution<double> distribution(-10.0, 10.0); // Range of random value

    // Generate random matrix A
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            A[i][j] = distribution(generator); // Assign a random double within the range
        }
    }

    // Proceed with LU decomposition and computing the L2,1 norm
    struct timeval start, end;
    long double diff;

    gettimeofday(&start,NULL);
    LU_Decomposition(nworkers, A, matrix_size, pi, L, U);
    gettimeofday(&end, NULL);
    diff = (end.tv_sec-start.tv_sec)*1000000 + end.tv_usec-start.tv_usec;

    printf("Time taken for LU decomposition: %Lf\n", diff);

    // Compute and print the L2,1 norm of the residual
    double L21Norm = computeL21Norm(nworkers, matrix_size, A, L, U, pi);
    cout << "L2,1 norm of the residual: " << L21Norm << endl;

    // Deallocate memory
    deallocateMatrix(A, matrix_size);
    deallocateMatrix_numa(L, matrix_size);
    deallocateMatrix_numa(U, matrix_size);

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