#include <iostream>
#include <chrono>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

#define Size 500

void multiplyMatrixNaive(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = 0;
            for (int k = 0; k < size; k++) {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

void multiplyMatrixCacheOptimized(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                result[k][j] = 0;
            for (int j = 0; j < size; j ++) {
                result[k][j] += m1[k][i] * m2[i][j];
            }
        }
    }
}

void multiplyMatrixCacheOptimizedUnrolled2(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    const int unroll = 2;  // unroll factor

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j += unroll) {
                result[k][j] = 0;
                result[k][j + 1] = 0;

                for (int u = 0; u < unroll; u++) {
                    result[k][j] += m1[k][i] * m2[i][j];
                    result[k][j + 1] += m1[k][i] * m2[i][j + 1];
                }
            }
        }
    }
}

void multiplyMatrixCacheOptimizedUnrolled4(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    const int unroll = 4;  // unroll factor

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j += unroll) {
                result[k][j] = 0;
                result[k][j + 1] = 0;
                result[k][j + 2] = 0;
                result[k][j + 3] = 0;

                for (int u = 0; u < unroll; u++) {
                    result[k][j] += m1[k][i] * m2[i][j];
                    result[k][j + 1] += m1[k][i] * m2[i][j + 1];
                    result[k][j + 2] += m1[k][i] * m2[i][j + 2];
                    result[k][j + 3] += m1[k][i] * m2[i][j + 3];
                }
            }
        }
    }
}

void multiplyMatrixCacheOptimizedUnrolled8(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    const int unroll = 8;  // unroll factor

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j += unroll) {
                result[k][j] = 0;
                result[k][j + 1] = 0;
                result[k][j + 2] = 0;
                result[k][j + 3] = 0;
                result[k][j + 4] = 0;
                result[k][j + 5] = 0;
                result[k][j + 6] = 0;
                result[k][j + 7] = 0;

                for (int u = 0; u < unroll; u++) {
                    result[k][j] += m1[k][i] * m2[i][j];
                    result[k][j + 1] += m1[k][i] * m2[i][j + 1];
                    result[k][j + 2] += m1[k][i] * m2[i][j + 2];
                    result[k][j + 3] += m1[k][i] * m2[i][j + 3];
                    result[k][j + 4] += m1[k][i] * m2[i][j + 4];
                    result[k][j + 5] += m1[k][i] * m2[i][j + 5];
                    result[k][j + 6] += m1[k][i] * m2[i][j + 6];
                    result[k][j + 7] += m1[k][i] * m2[i][j + 7];
                }
            }
        }
    }
}


void multiplyMatrixNeonARM(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    #pragma omp parallel for
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                result[k][j] = 0;
            for (int j = 0; j < size; j ++) {
                // IMPLEMENTARE NEON ARM
                int32x4_t v1 = vld1q_s32(&m1[k][i]);
                int32x4_t v2 = vld1q_s32(&m2[i][j]);
                int32x4_t v_result = vmulq_s32(v1, v2);
                int32_t result_lane[4];
                vst1q_s32(result_lane, v_result);
                result[k][j] = result_lane[0] + result_lane[1] + result_lane[2] + result_lane[3];
            }
        }
    }
}
//testing. other implementations
// void multiplyMatrixNEONARM(int32_t *A, int32_t *B, int32_t *C, uint32_t size) {
//     float32x4_t A_vector, B_vector, C_vector;

//     for (uint32_t i = 0; i < size; i++) {
//         for (uint32_t j = 0; j < size; j++) {
//             C_vector = vmovq_n_f32(0);
//             for (uint32_t k = 0; k < size; k++) {
//                 A_vector = vld1q_f32(A + i * size + k);
//                 B_vector = vld1q_f32(B + k * size + j);
//                 C_vector = vfmaq_f32(C_vector, A_vector, B_vector);
//             }
//             vst1q_f32(C + i * size + j, C_vector);
//         }
//     }
// }

// void multiplyMatrixNEONARM()
// {
//     float32x4x4_t myMat;
//     float32x2_t myVecLow, myVecHigh;

//     myVecLow = vld1_f32(&pVec[0]);
//     myVecHigh = vld1_f32(&pVec[2]);
//     myMat = vld4q_f32(pMat);

//     myMat.val[0] = vmulq_lane_f32(myMat.val[0], myVecLow, 0);
//     myMat.val[0] = vmlaq_lane_f32(myMat.val[0], myMat.val[1], myVecLow, 1);
//     myMat.val[0] = vmlaq_lane_f32(myMat.val[0], myMat.val[2], myVecHigh, 0);
//     myMat.val[0] = vmlaq_lane_f32(myMat.val[0], myMat.val[3], myVecHigh, 1);

//     vst1q_f32(pDst, myMat.val[0]);
// }

void multiplyMatrixNEON(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i+=4) {
        for (int j = 0; j < size; j+=4) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            for (int k = 0; k < size; k+=4) {
                // load
                int32x4_t v1 = vld1q_s32(m1[i]);
                int32x4_t v2 = vld1q_s32(m2[k]+j);
                acc0 = vmlaq_laneq_s32(acc0, v1, v2, 0);
                acc1 = vmlaq_laneq_s32(acc1, v1, v2, 1);
                acc2 = vmlaq_laneq_s32(acc2, v1, v2, 2);
                acc3 = vmlaq_laneq_s32(acc3, v1, v2, 3);
            }
            // store
            result[i][j] = vaddvq_s32(acc0);
            result[i][j+1] = vaddvq_s32(acc1);
            result[i][j+2] = vaddvq_s32(acc2);
            result[i][j+3] = vaddvq_s32(acc3);
        }
    }
}

void multiplyMatrixCacheParalel(int32_t** m1, int32_t** m2, int32_t** result, int size) {
    #pragma omp parallel for
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                result[k][j] = 0;
            for (int j = 0; j < size; j ++) {
                result[k][j] += m1[k][i] * m2[i][j];
            }
        }
    }
}

int main() {
    omp_set_num_threads(4);
    int dimensions[] = {200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};
    int numDimensions = sizeof(dimensions) / sizeof(dimensions[0]);

    for (int dimIndex = 0; dimIndex < numDimensions; dimIndex++) {
        int dim = dimensions[dimIndex];
        int32_t** m1 = new int32_t*[dim];
        int32_t** m2 = new int32_t*[dim];
        int32_t** result = new int32_t*[dim];

        for (int i = 0; i < dim; i++) {
            m1[i] = new int32_t[dim];
            m2[i] = new int32_t[dim];
            result[i] = new int32_t[dim];
        }

        auto clock_start0 = chrono::high_resolution_clock::now();
        multiplyMatrixNaive(m1, m2, result, dim);
        auto clock_end0 = chrono::high_resolution_clock::now();
        cout << "Timpul de executie pentru dimensiunea " << dim << " (NAIVE): " << chrono::duration_cast<chrono::milliseconds>(clock_end0 - clock_start0).count() << " milisecunde" << endl;

        auto clock_start1 = chrono::high_resolution_clock::now();
        multiplyMatrixCacheOptimized(m1, m2, result, dim);
        auto clock_end1 = chrono::high_resolution_clock::now();
        cout<< "Timpul de executie pentru dimensiunea " << dim << " (CACHE-OPTIMIZED): " << chrono::duration_cast<chrono::milliseconds>(clock_end1 - clock_start1).count() << " milisecunde" << endl;

        auto clock_start2 = chrono::high_resolution_clock::now();
        multiplyMatrixNeonARM(m1, m2, result, dim);
        auto clock_end2 = chrono::high_resolution_clock::now();
        cout << "Timpul de executie pentru dimensiunea " << dim << " (NEON-ARM): " << chrono::duration_cast<chrono::milliseconds>(clock_end2 - clock_start2).count() << " milisecunde" << endl;

        auto clock_start3 = chrono::high_resolution_clock::now();
        multiplyMatrixNEON(m1, m2, result, dim);
        auto clock_end3 = chrono::high_resolution_clock::now();
        cout << "Timpul de executie pentru dimensiunea " << dim << " (NEON): " << chrono::duration_cast<chrono::milliseconds>(clock_end3 - clock_start3).count() << " milisecunde" << endl;

        auto clock_start4 = chrono::high_resolution_clock::now();
        multiplyMatrixCacheParalel(m1, m2, result, dim);
        auto clock_end4 = chrono::high_resolution_clock::now();
        cout << "Timpul de executie pentru dimensiunea " << dim << " (CACHE-PARALEL): " << chrono::duration_cast<chrono::milliseconds>(clock_end4 - clock_start4).count() << " milisecunde" << endl << endl;

        for (int i = 0; i < dim; i++) {
            delete[] m1[i];
            delete[] m2[i];
            delete[] result[i];
        }
        delete[] m1;
        delete[] m2;
        delete[] result;
    }

    return 0;
}

// For testing unrolled functions
// int main() {
//     int32_t** m1 = new int32_t*[Size];
//     int32_t** m2 = new int32_t*[Size];
//     int32_t** result = new int32_t*[Size];

//     for (int i = 0; i < Size; i++) {
//         m1[i] = new int32_t[Size];
//         m2[i] = new int32_t[Size];
//         result[i] = new int32_t[Size];
//     }

//     auto clock_start1 = chrono::high_resolution_clock::now();
//     multiplyMatrixNaive(m1, m2, result, Size);
//     auto clock_end1 = chrono::high_resolution_clock::now();
//     cout << "Timpul de executie pentru prima varianta (Naive): " << chrono::duration_cast<chrono::milliseconds>(clock_end1 - clock_start1).count() << " milisecunde" << endl;

//     auto clock_start2 = chrono::high_resolution_clock::now();
//     multiplyMatrixCacheOptimized(m1, m2, result, Size);
//     auto clock_end2 = chrono::high_resolution_clock::now();
//     cout << "Timpul de executie pentru a doua varianta: " << chrono::duration_cast<chrono::milliseconds>(clock_end2 - clock_start2).count() << " milisecunde" << endl;

//     auto clock_start3 = chrono::high_resolution_clock::now();
//     multiplyMatrixCacheOptimizedUnrolled2(m1, m2, result, Size);
//     auto clock_end3 = chrono::high_resolution_clock::now();
//     cout << "Timpul de executie pentru multiplyMatrixUnrolled2: " << chrono::duration_cast<chrono::milliseconds>(clock_end3 - clock_start3).count() << " milisecunde" << endl;

//     auto clock_start4 = chrono::high_resolution_clock::now();
//     multiplyMatrixCacheOptimizedUnrolled4(m1, m2, result, Size);
//     auto clock_end4 = chrono::high_resolution_clock::now();
//     cout << "Timpul de executie pentru multiplyMatrixUnrolled4: " << chrono::duration_cast<chrono::milliseconds>(clock_end4 - clock_start4).count() << " milisecunde" << endl;

//     auto clock_start5 = chrono::high_resolution_clock::now();
//     multiplyMatrixCacheOptimizedUnrolled8(m1, m2, result, Size);
//     auto clock_end5 = chrono::high_resolution_clock::now();
//     cout << "Timpul de executie pentru multiplyMatrixUnrolled8: " << chrono::duration_cast<chrono::milliseconds>(clock_end5 - clock_start5).count() << " milisecunde" << endl;

//     for (int i = 0; i < Size; i++) {
//         delete[] m1[i];
//         delete[] m2[i];
//         delete[] result[i];
//     }
//     delete[] m1;
//     delete[] m2;
//     delete[] result;

//     return 0;
// }
