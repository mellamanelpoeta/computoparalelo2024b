#include <iostream>
#include <vector>
#include <omp.h>

// Define the size of the vectors
long int SIZE = 1000000000;

// Serial version: sum_vectors_serial
void sum_vectors_serial(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size) {
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}

// Parallel version: sum_vectors_parallel using OpenMP parallel sections
void sum_vectors_parallel(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int size) {
    #pragma omp parallel sections
    {
        // First section: process the first half of the vectors
        #pragma omp section
        {
            for (int i = 0; i < size / 2; ++i) {
                C[i] = A[i] + B[i];
            }
        }

        // Second section: process the second half of the vectors
        #pragma omp section
        {
            for (int i = size / 2; i < size; ++i) {
                C[i] = A[i] + B[i];
            }
        }
    }
}

int main() {
    // Allocate and initialize large vectors A and B
    std::vector<double> A(SIZE);
    std::vector<double> B(SIZE);
    std::vector<double> C_serial(SIZE);   // Result vector for serial computation
    std::vector<double> C_parallel(SIZE); // Result vector for parallel computation

    // Initialize vectors A and B in parallel for efficiency
    #pragma omp parallel for
    for (int i = 0; i < SIZE; ++i) {
        A[i] = static_cast<double>(i) * 1.0;
        B[i] = static_cast<double>(i) * 2.0;
    }

    // =========================
    // Serial Computation
    // =========================
    double start_time_serial = omp_get_wtime();
    sum_vectors_serial(A, B, C_serial, SIZE);
    double end_time_serial = omp_get_wtime();
    double time_serial = end_time_serial - start_time_serial;

    // =========================
    // Parallel Computation
    // =========================
    double start_time_parallel = omp_get_wtime();
    sum_vectors_parallel(A, B, C_parallel, SIZE);
    double end_time_parallel = omp_get_wtime();
    double time_parallel = end_time_parallel - start_time_parallel;

    // =========================
    // Verify Correctness
    // =========================
    bool correct = true;
    for (int i = 0; i < SIZE; ++i) {
        if (C_serial[i] != C_parallel[i]) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << "C_serial[" << i << "] = " << C_serial[i]
                      << ", C_parallel[" << i << "] = " << C_parallel[i] << std::endl;
            break;
        }
    }

    if (correct) {
        std::cout << "Both serial and parallel computations are correct." << std::endl;
    } else {
        std::cerr << "There is a mismatch between serial and parallel computations." << std::endl;
    }

    // =========================
    // Calculate Speedup
    // =========================
    double speedup = time_serial / time_parallel;

    // =========================
    // Display Results
    // =========================
    std::cout << "Serial Execution Time   : " << time_serial << " seconds" << std::endl;
    std::cout << "Parallel Execution Time : " << time_parallel << " seconds" << std::endl;
    std::cout << "Speedup                 : " << speedup << "x" << std::endl;

    // Optionally print some values to check correctness
    std::cout << "C_serial[0] = " << C_serial[0] 
              << ", C_serial[" << SIZE - 1 << "] = " << C_serial[SIZE - 1] << std::endl;
    std::cout << "C_parallel[0] = " << C_parallel[0] 
              << ", C_parallel[" << SIZE - 1 << "] = " << C_parallel[SIZE - 1] << std::endl;

    return 0;
}
