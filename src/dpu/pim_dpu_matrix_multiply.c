#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mram.h>
#include <alloc.h>
#include <defs.h>
#include <barrier.h>

#include "pim_dpu_matrix_multiply_thread_memory_manager.h"

#include "dpu_pim_matrix_multiply_kernel_arguments.h"


__host dpu_pim_matrix_multiply_kernel_arguments_t MATRIX_MULTIPLY_ARGUMENTS;
__dma_aligned void* aux;

BARRIER_INIT(my_barrier, NR_TASKLETS);

/**
 * @brief Main DPU function for matrix multiplication
 * 
 * This function is executed on each DPU and coordinates tasklets to perform
 * parallel matrix multiplication using the memory manager.
 */
int main() {
    int pid = me();
    
    // Initialize memory heap on tasklet 0
    if (pid == 0) {
        mem_reset(); // Reset the heap
    }
    barrier_wait(&my_barrier);
    
    // Extract arguments from host
    uint32_t matrix1_start_offset = MATRIX_MULTIPLY_ARGUMENTS.matrix1_start_offset;
    uint32_t matrix2_start_offset = MATRIX_MULTIPLY_ARGUMENTS.matrix2_start_offset;
    uint32_t result_start_offset = MATRIX_MULTIPLY_ARGUMENTS.result_start_offset;
    uint32_t matrix1_type_size = MATRIX_MULTIPLY_ARGUMENTS.matrix1_type_size;
    uint32_t matrix2_type_size = MATRIX_MULTIPLY_ARGUMENTS.matrix2_type_size;
    uint32_t result_type_size = MATRIX_MULTIPLY_ARGUMENTS.result_type_size;
    
    uint32_t matrix1_rows = MATRIX_MULTIPLY_ARGUMENTS.matrix1_rows;
    uint32_t matrix1_cols = MATRIX_MULTIPLY_ARGUMENTS.matrix1_cols;
    uint32_t matrix2_rows = MATRIX_MULTIPLY_ARGUMENTS.matrix2_rows;
    uint32_t matrix2_cols = MATRIX_MULTIPLY_ARGUMENTS.matrix2_cols;
    uint32_t result_rows = MATRIX_MULTIPLY_ARGUMENTS.result_rows;
    uint32_t result_cols = MATRIX_MULTIPLY_ARGUMENTS.result_cols;
    
    // DEBUG: Only tasklet 0 reads and prints the matrices
    if (pid == 0) {
        printf("=== DEBUG: Reading matrices from MRAM ===\n");
        printf("Matrix1: %ux%u, Matrix2: %ux%u, Result: %ux%u\n",
               matrix1_rows, matrix1_cols, matrix2_rows, matrix2_cols, result_rows, result_cols);
        printf("MRAM offsets: M1=%u, M2=%u, Result=%u\n",
               matrix1_start_offset, matrix2_start_offset, result_start_offset);
        
        // Calculate matrix sizes
        uint32_t matrix1_size = matrix1_rows * matrix1_cols * matrix1_type_size;
        uint32_t matrix2_size = matrix2_rows * matrix2_cols * matrix2_type_size;
        
        // Ensure 8-byte alignment for MRAM transfers
        uint32_t aligned_matrix1_size = ((matrix1_size + 8 - (matrix1_size % 8)));
        uint32_t aligned_matrix2_size = ((matrix2_size + 8 - (matrix1_size % 8)));
        
        printf("Matrix sizes: M1=%u bytes (aligned=%u), M2=%u bytes (aligned=%u)\n",
               matrix1_size, aligned_matrix1_size, matrix2_size, aligned_matrix2_size);
        // Allocate WRAM for both matrices using mem_alloc
        uint8_t* matrix1_wram = (uint8_t*)mem_alloc(aligned_matrix1_size);
        uint8_t* matrix2_wram = (uint8_t*)mem_alloc(aligned_matrix2_size);

        if (matrix1_wram == NULL || matrix2_wram == NULL) {
            printf("ERROR: Failed to allocate WRAM for debug matrices\n");
            return -1;
        }
        
        printf("WRAM allocated: M1=%p, M2=%p\n", matrix1_wram, matrix2_wram);
        
        // Read Matrix1 from MRAM to WRAM
        __mram_ptr void* matrix1_mram = DPU_MRAM_HEAP_POINTER + matrix1_start_offset;
        printf("Reading Matrix1 from MRAM address %p...\n", matrix1_mram);
        mram_read(matrix1_mram, matrix1_wram, aligned_matrix1_size);
        
        // Read Matrix2 from MRAM to WRAM
        __mram_ptr void* matrix2_mram = DPU_MRAM_HEAP_POINTER + matrix2_start_offset;
        printf("Reading Matrix2 from MRAM address %p...\n", matrix2_mram);
        mram_read(matrix2_mram, matrix2_wram, aligned_matrix2_size);
        
        printf("=== Matrix1 Contents ===\n");
        for (uint32_t i = 0; i < matrix1_rows && i < 8; i++) { // Limit rows for readability
            printf("Row %u: ", i);
            for (uint32_t j = 0; j < matrix1_cols && j < 16; j++) { // Limit cols for readability
                uint32_t idx = i * matrix1_cols + j;
                printf("%02x ", matrix1_wram[idx]);
            }
            if (matrix1_cols > 16) printf("... (%u more cols)", matrix1_cols - 16);
            printf("\n");
        }
        if (matrix1_rows > 8) printf("... (%u more rows)\n", matrix1_rows - 8);
        
        printf("\n=== Matrix2 Contents ===\n");
        for (uint32_t i = 0; i < matrix2_rows && i < 8; i++) { // Limit rows for readability
            printf("Row %u: ", i);
            for (uint32_t j = 0; j < matrix2_cols && j < 16; j++) { // Limit cols for readability
                uint32_t idx = i * matrix2_cols + j;
                printf("%02x ", matrix2_wram[idx]);
            }
            if (matrix2_cols > 16) printf("... (%u more cols)", matrix2_cols - 16);
            printf("\n");
        }
        if (matrix2_rows > 8) printf("... (%u more rows)\n", matrix2_rows - 8);
        
        printf("=== End Debug Output ===\n\n");

        // Naive matrix multiplication (assuming uint8_t elements for simplicity)
        if (matrix1_cols != matrix2_cols) {
            printf("ERROR: Incompatible matrix dimensions for multiplication\n");
        } else {
            uint32_t result_size = result_rows * result_cols * result_type_size;
            uint32_t aligned_result_size = ((result_size + 8 - (result_size % 8)));
            uint16_t* result_wram = (uint16_t*)mem_alloc(aligned_result_size);
            if (result_wram == NULL) {
            printf("ERROR: Failed to allocate WRAM for result matrix\n");
            } else {
            // Zero the result matrix
            for (uint32_t i = 0; i < aligned_result_size; i++) {
                result_wram[i] = 0;
            }
            // Naive multiplication
            for (uint32_t i = 0; i < matrix1_rows; i++) {
                for (uint32_t j = 0; j < matrix2_cols; j++) {
                uint32_t sum = 0;
                for (uint32_t k = 0; k < matrix1_cols; k++) {
                    uint8_t a = matrix1_wram[i * matrix1_cols + k];
                    uint8_t b = matrix2_wram[j * matrix1_cols + k];
                    sum += a * b;
                }
                result_wram[j * matrix1_rows + i] = (uint16_t)sum;
                }
            }
            printf("\n=== Naive Result Matrix ===\n");
            for (uint32_t i = 0; i < result_rows && i < 8; i++) {
                printf("Row %u: ", i);
                for (uint32_t j = 0; j < result_cols && j < 16; j++) {
                uint32_t idx = i * result_cols + j;
                printf("%02x ", result_wram[idx]);
                }
                if (result_cols > 16) printf("... (%u more cols)", result_cols - 16);
                printf("\n");
            }
            if (result_rows > 8) printf("... (%u more rows)\n", result_rows - 8);
            printf("=== End Naive Result ===\n\n");

            // Write result matrix back to MRAM
            __mram_ptr void* result_mram = DPU_MRAM_HEAP_POINTER + result_start_offset;
            printf("Writing result matrix to MRAM address %p...\n", result_mram);
            mram_write(result_wram, result_mram, aligned_result_size);
            }
        }
    }

    // Wait for debug output to complete
    barrier_wait(&my_barrier);
    
    // Create matrix configuration
    matrix_config_t config = {
        .matrix1_rows = matrix1_rows,
        .matrix1_cols = matrix1_cols,
        .matrix2_rows = matrix2_rows,
        .matrix2_cols = matrix2_cols,
        .result_rows = result_rows,
        .result_cols = result_cols,
        .tasklet_id = pid,
        .num_tasklets = NR_TASKLETS
    };
    
    // Call the memory manager to handle matrix multiplication
    // return pim_dpu_matrix_multiply_thread_memory_manager(
    //     DPU_MRAM_HEAP_POINTER + matrix1_start_offset,
    //     DPU_MRAM_HEAP_POINTER + matrix2_start_offset,
    //     DPU_MRAM_HEAP_POINTER + result_start_offset,
    //     matrix1_type_size,
    //     matrix2_type_size,
    //     result_type_size,
    //     &config
    // );
}
