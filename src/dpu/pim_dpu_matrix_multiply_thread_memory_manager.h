#ifndef __PIM_DPU_MATRIX_MULTIPLY_THREAD_MEMORY_MANAGER_H__
#define __PIM_DPU_MATRIX_MULTIPLY_THREAD_MEMORY_MANAGER_H__

#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <mutex.h>

#define MAX_MATRIX_ROWS 512
#define MAX_MATRIX_COLS 512

/**
 * @brief Matrix configuration structure for DPU operations
 */
typedef struct {
    uint32_t matrix1_rows;
    uint32_t matrix1_cols;
    uint32_t matrix2_rows;
    uint32_t matrix2_cols;
    uint32_t result_rows;
    uint32_t result_cols;
    uint32_t tasklet_id;
    uint32_t num_tasklets;
} matrix_config_t;


uint16_t pim_dpu_dot_product(uint8_t * first_matrix, uint8_t * second_matrix, int32_t elements) {
    // Input validation
    if (first_matrix == NULL || second_matrix == NULL || elements <= 0) {
        printf("Error: Null matrix pointer provided.\n");
        return 0;
    }
    
    int16_t result = 0;
    for (uint32_t i = 0; i < elements; ++i) {
        result += (int16_t)first_matrix[i] * (int16_t)second_matrix[i];
    }
    return result;
}

BARRIER_INIT(barrier_p, NR_TASKLETS);



// Global shared WRAM data structures
static __dma_aligned uint8_t* global_matrix1_rows[MAX_MATRIX_ROWS];  // Array of pointers to WRAM row data
static __dma_aligned uint8_t* global_matrix2_cols[MAX_MATRIX_COLS];  // Array of pointers to WRAM column data
static __dma_aligned uint16_t* global_result_matrix;                 // WRAM result matrix
static bool global_matrix1_row_fetched[MAX_MATRIX_ROWS];            // Track which rows are fetched
static bool global_matrix2_col_fetched[MAX_MATRIX_COLS];            // Track which columns are fetched

// FSB allocators for memory management
static fsb_allocator_t global_result_allocator;
static fsb_allocator_t global_matrix1_row_allocators[MAX_MATRIX_ROWS];
static fsb_allocator_t global_matrix2_col_allocators[MAX_MATRIX_COLS];

// Mutexes for thread coordination
MUTEX_INIT(matrix1_mutex);  // Protects matrix1 row fetching
MUTEX_INIT(matrix2_mutex);  // Protects matrix2 column fetching
MUTEX_INIT(result_mutex);   // Protects result matrix writes
MUTEX_INIT(log_mutex);     // Protects debug logging

#define MAX_MATRIX_ROWS 512
#define MAX_MATRIX_COLS 512

/**
 * @brief Memory manager for DPU matrix multiplication with tasklet distribution
 * 
 * This function manages memory allocation and data distribution across tasklets for 
 * matrix multiplication operations. It follows SimplePIM's MapProcessing pattern:
 * 1. Splits matrices into tasklet-sized chunks
 * 2. Manages MRAM to WRAM transfers
 * 3. Coordinates parallel processing across tasklets
 * 4. Handles memory barriers and synchronization
 *
 * @param inputs1 Pointer to first matrix data in MRAM
 * @param inputs2 Pointer to second matrix data in MRAM  
 * @param outputs Pointer to result matrix data in MRAM
 * @param input_type1 Size of first matrix elements (int8_t = 1)
 * @param input_type2 Size of second matrix elements (int8_t = 1)
 * @param output_type Size of result matrix elements (uint16_t = 2)
 * @param config Matrix dimensions and tasklet configuration
 */
int pim_dpu_matrix_multiply_thread_memory_manager(__mram_ptr void* inputs1, __mram_ptr void* inputs2, __mram_ptr void* outputs, uint32_t input_type1, uint32_t input_type2, uint32_t output_type, matrix_config_t* config) {
    uint32_t pid = me();
    uint32_t num_tasklets = NR_TASKLETS;
    
    // Matrix dimensions from config
    uint32_t matrix1_rows = config->matrix1_rows;
    uint32_t matrix1_cols = config->matrix1_cols;
    uint32_t matrix2_rows = config->matrix2_rows;
    uint32_t matrix2_cols = config->matrix2_cols;
    uint32_t result_rows = config->result_rows;
    uint32_t result_cols = config->result_cols;
    
    // Validate matrix dimensions for multiplication
    if (matrix1_cols != matrix2_rows) {
        if (pid == 0) {
            printf("Error: Matrix dimensions incompatible for multiplication (matrix1_cols=%u != matrix2_rows=%u)\n", 
                   matrix1_cols, matrix2_rows);
        }
        return -1;
    }
    
    // Check if matrices exceed maximum dimensions
    if (matrix1_rows > MAX_MATRIX_ROWS || matrix2_cols > MAX_MATRIX_COLS) {
        if (pid == 0) {
            printf("Error: Matrix dimensions exceed maximum (max_rows=%u, max_cols=%u)\n", 
                   MAX_MATRIX_ROWS, MAX_MATRIX_COLS);
        }
        return -1;
    }
    
    // Calculate total result elements and check if we have too many threads
    uint32_t total_result_elements = result_rows * result_cols;
    if (num_tasklets > total_result_elements) {
        if (pid == 0) {
            printf("Error: Too many threads (%u) for result elements (%u)\n", 
                   num_tasklets, total_result_elements);
        }
        return -1;
    }
    
    // Thread 0 initializes WRAM address arrays and result matrix
    if (pid == 0) {
        printf("Thread 0: Initializing WRAM address arrays and result matrix\n");
        printf("Thread 0: MRAM pointers - inputs1=%p, inputs2=%p, outputs=%p\n", 
               inputs1, inputs2, outputs);
        printf("Thread 0: Matrix dimensions - m1: %ux%u, m2: %ux%u, result: %ux%u\n",
               matrix1_rows, matrix1_cols, matrix2_rows, matrix2_cols, result_rows, result_cols);
        
        // Initialize all row and column pointers to NULL
        for (uint32_t i = 0; i < MAX_MATRIX_ROWS; i++) {
            global_matrix1_rows[i] = NULL;
            global_matrix1_row_fetched[i] = false;
        }
        
        for (uint32_t i = 0; i < MAX_MATRIX_COLS; i++) {
            global_matrix2_cols[i] = NULL;
            global_matrix2_col_fetched[i] = false;
        }
        
        // Allocate WRAM for result matrix using FSB allocator
        uint32_t result_size = result_rows * result_cols * output_type;
        global_result_allocator = fsb_alloc(result_size, 1);
        global_result_matrix = (uint16_t*)fsb_get(global_result_allocator);
        if (global_result_matrix == NULL) {
            printf("Error: Failed to allocate WRAM for result matrix (%u bytes)\n", result_size);
            return -1;
        }
        
        // Initialize result matrix to zero
        for (uint32_t i = 0; i < result_rows * result_cols; i++) {
            global_result_matrix[i] = 0;
        }
        
        printf("Thread 0: Allocated result matrix in WRAM (%u bytes)\n", result_size);
    }
    
    // Wait for thread 0 to complete initialization
    barrier_wait(&barrier_p);
    
    // Calculate which result elements this thread should compute
    uint32_t elements_per_thread = total_result_elements / num_tasklets;
    uint32_t extra_elements = total_result_elements % num_tasklets;
    
    uint32_t start_element, end_element;
    if (pid < extra_elements) {
        // Threads with extra elements
        start_element = pid * (elements_per_thread + 1);
        end_element = start_element + elements_per_thread + 1;
    } else {
        // Regular threads
        start_element = extra_elements * (elements_per_thread + 1) + (pid - extra_elements) * elements_per_thread;
        end_element = start_element + elements_per_thread;
    }
    
    // Process assigned result elements
    for (uint32_t elem_idx = start_element; elem_idx < end_element; elem_idx++) {
        // Convert linear element index to matrix coordinates
        uint32_t result_row = elem_idx / result_cols;
        uint32_t result_col = elem_idx % result_cols;
        
        // Ensure matrix1 row is fetched
        mutex_lock(matrix1_mutex);
        if (!global_matrix1_row_fetched[result_row]) {
            // Allocate WRAM for this row using FSB allocator (aligned)
            uint32_t row_size = matrix1_cols * input_type1;
            uint32_t aligned_row_size = ((row_size + 7) / 8) * 8; // 8-byte alignment
            global_matrix1_row_allocators[result_row] = fsb_alloc(aligned_row_size, 1);
            global_matrix1_rows[result_row] = (uint8_t*)fsb_get(global_matrix1_row_allocators[result_row]);
            if (global_matrix1_rows[result_row] == NULL) {
                printf("Thread %u: Failed to allocate WRAM for matrix1 row %u (size=%u)\n", 
                       pid, result_row, aligned_row_size);
                mutex_unlock(matrix1_mutex);
                return -1;
            }
            
            // Fetch row from MRAM (aligned transfer following SimplePIM pattern)
            uint32_t mram_offset = result_row * matrix1_cols * input_type1;
            mram_read((__mram_ptr void*)((char*)inputs1 + mram_offset), global_matrix1_rows[result_row], aligned_row_size);
            global_matrix1_row_fetched[result_row] = true;
        }
        mutex_unlock(matrix1_mutex);
        
        // Ensure matrix2 column is fetched
        mutex_lock(matrix2_mutex);
        if (!global_matrix2_col_fetched[result_col]) {
            // Allocate WRAM for this column using FSB allocator
            uint32_t col_size = matrix2_rows * input_type2;
            global_matrix2_col_allocators[result_col] = fsb_alloc(col_size, 1);
            global_matrix2_cols[result_col] = (uint8_t*)fsb_get(global_matrix2_col_allocators[result_col]);
            if (global_matrix2_cols[result_col] == NULL) {
                printf("Thread %u: Failed to allocate WRAM for matrix2 column %u\n", pid, result_col);
                mutex_unlock(matrix2_mutex);
                return -1;
            }
            
            // Fetch column from MRAM (optimized bulk transfer)
            // Since matrix2 is stored row-major, we need to gather column elements
            // This is less efficient but necessary for column access
            // For column-major ordering, we can copy the entire column in one go
            uint8_t* temp_col_buffer = global_matrix2_cols[result_col];
            uint32_t mram_col_offset = result_col * matrix2_rows * input_type2;
            mram_read((__mram_ptr void*)((char*)inputs2 + mram_col_offset), global_matrix2_cols[result_col], col_size);
            global_matrix2_col_fetched[result_col] = true;
        }
        mutex_unlock(matrix2_mutex);
        
        // Calculate dot product for result[result_row][result_col]
        uint16_t dot_product = 0;
        uint8_t* row_data = global_matrix1_rows[result_row];
        uint8_t* col_data = global_matrix2_cols[result_col];
        
        for (uint32_t k = 0; k < matrix1_cols; k++) {
            uint8_t a_val = row_data[k];
            uint8_t b_val = col_data[k];
            dot_product += a_val * b_val;
        }
        
        // Store result in WRAM result matrix
        mutex_lock(result_mutex);
        global_result_matrix[elem_idx] = dot_product;
        mutex_unlock(result_mutex);
    }
    
    // Wait for all threads to complete their calculations
    barrier_wait(&barrier_p);
    
    // Thread 0 writes the result back to MRAM
    if (pid == 0) {
        printf("Thread 0: Writing result matrix back to MRAM...\n");
        uint32_t result_size = result_rows * result_cols * output_type;
        mram_write(global_result_matrix, outputs, result_size);
        printf("Thread 0: Completed writing %u result elements (%u bytes)\n", 
               result_rows * result_cols, result_size);
        
        // Free allocated WRAM memory using FSB allocators
        fsb_free(global_result_allocator, global_result_matrix);
        for (uint32_t i = 0; i < matrix1_rows; i++) {
            if (global_matrix1_rows[i] != NULL) {
                fsb_free(global_matrix1_row_allocators[i], global_matrix1_rows[i]);
            }
        }
        for (uint32_t i = 0; i < matrix2_cols; i++) {
            if (global_matrix2_cols[i] != NULL) {
                fsb_free(global_matrix2_col_allocators[i], global_matrix2_cols[i]);
            }
        }
        printf("Thread 0: Freed all allocated WRAM memory\n");
    }
    
    // Final synchronization
    barrier_wait(&barrier_p);
    return 0;
}

#endif // __PIM_DPU_MATRIX_MULTIPLY_THREAD_MEMORY_MANAGER_H__