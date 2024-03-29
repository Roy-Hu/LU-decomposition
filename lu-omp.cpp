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
void deallocateMatrix_numa(int nworkers, double** matrix, int n) {
    #pragma omp parallel for num_threads(nworkers) schedule(static, 1) shared(n, matrix, nworkers) default(none)
    for (int i = 0; i < n; ++i) {
        numa_free(matrix[i], sizeof(double) * n);
    }
    delete[] matrix;
}
// Function to compute the product of matrices L and U
double** multiplyLU(double** L, double** U, int n) {
    double** LU = allocateMatrix(n);

    #pragma omp parallel for num_threads(16)
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

    #pragma omp parallel for num_threads(16)
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
    omp_set_num_threads(16);

    double** PA = multiplyPA(A, P, n);
    double** LU = multiplyLU( L, U, n);

    // Compute the residual R = PA - LU
    double** R = allocateMatrix(n);


    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            R[i][j] = PA[i][j] - LU[i][j];
        }
    }

    // Compute the L2,1 norm of the residual
    // parallelize with reduction
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
    MaxKPrime max_k_prime = {0.0, -1};

    #pragma omp parallel num_threads(nworkers) shared(n, L, U, A, pi, A_prime, max_k_prime, nworkers) default(none)
    {
        // Allocate memory for matrices L, U, and A_prime, and pi
        // Use static, 1 scheduling to distribute the work among threads
        // Use reduction to find the maximum value of A_prime[k][0] and its index k
        #pragma omp for schedule(static, 1) reduction(maximum: max_k_prime) 
        for (int i = 0; i < n; ++i) {
            L[i] = (double *)numa_alloc_local(n * sizeof(double));
            U[i] = (double *)numa_alloc_local(n * sizeof(double));
            A_prime[i] = (double *)numa_alloc_local(n * sizeof(double));

            memset(U[i], 0, n * sizeof(double));
            memset(L[i], 0, n * sizeof(double));
            memcpy(A_prime[i], A[i], n * sizeof(double));
            
            L[i][i] = 1;
            pi[i] = i;

            double abs_val = abs(A_prime[i][0]);
            if (abs_val > max_k_prime.max) {
                max_k_prime.max = abs_val;
                max_k_prime.k_prime = i;
            }
        }

        // implicit barrier

        for (int k = 0; k < n; ++k) {
            // The swapping for each metrix is done by different threads
            #pragma omp sections
            {
                #pragma omp section
                {   
                    swap(pi[k], pi[max_k_prime.k_prime]);
                }

                #pragma omp section
                {   
                    double tmp[k];
                    memcpy(tmp, L[k], k * sizeof(double));
                    memcpy(L[k], L[max_k_prime.k_prime], k * sizeof(double));
                    memcpy(L[max_k_prime.k_prime], tmp, k * sizeof(double));
                }

                #pragma omp section
                {   
                    memcpy(U[k] + k, A_prime[max_k_prime.k_prime] + k, (n - k) * sizeof(double));
                    // A_prime[k] is not be use in the future, don't need to swap
                    memcpy(A_prime[max_k_prime.k_prime] + k, A_prime[k] + k, (n - k) * sizeof(double));
                }
            }

            // implicit barrier

            #pragma omp single
            {
                max_k_prime = {0.0, -1};
            }

            // implicit barrier

            // (k + 1) - (k + 1) % nworkers to allign for thread and data's numa node            
            // Use static, 1 to match the row's numa node to the thread
            #pragma omp for schedule(static, 1) reduction(maximum: max_k_prime) 
            for (int i = (k + 1) - (k + 1) % nworkers; i < n; ++i) {
                if (i < k + 1) continue;
                L[i][k] = A_prime[i][k] / U[k][k];
                
                // Use simd to vectorize the loop
                #pragma omp simd
                for (int j = k + 1; j < n; ++j) {
                    A_prime[i][j] -= L[i][k] * U[k][j];
                }

                double abs_val = abs(A_prime[i][k + 1]);
                if (abs_val > max_k_prime.max) {
                    max_k_prime.max = abs_val;
                    max_k_prime.k_prime = i;
                }
            }
            // implicit barrier
        }
    }

    deallocateMatrix_numa(nworkers, A_prime, n);
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
    omp_set_max_active_levels(2);

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
    mt19937 generator(42); // Random number generator seeded with current time
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
    deallocateMatrix_numa(nworkers, L, matrix_size);
    deallocateMatrix_numa(nworkers, U, matrix_size);

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
