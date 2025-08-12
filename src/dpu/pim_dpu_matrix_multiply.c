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
    return pim_dpu_matrix_multiply_thread_memory_manager(
        DPU_MRAM_HEAP_POINTER + matrix1_start_offset,
        DPU_MRAM_HEAP_POINTER + matrix2_start_offset,
        DPU_MRAM_HEAP_POINTER + result_start_offset,
        matrix1_type_size,
        matrix2_type_size,
        result_type_size,
        &config
    );
}
