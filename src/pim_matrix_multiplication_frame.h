#ifndef __PIM_MATRIX_MULTIPLICATION_FRAME_H___
#define __PIM_MATRIX_MULTIPLICATION_FRAME_H___

#include <dpu.h>

typedef struct {
    uint32_t num_work_groups;
    uint32_t work_group_size;
    uint32_t num_dpus;
    uint32_t matrix1_rows;
    uint32_t matrix1_cols;
    uint32_t matrix2_rows;
    uint32_t matrix2_cols;
    uint32_t result_rows;
    uint32_t result_cols;
    uint32_t matrix1_type_size;       ///< Size of first matrix elements (1 for
    uint32_t matrix2_type_size;       ///< Size of second matrix elements (1 for
    uint32_t result_type_size;        ///< Size of result matrix elements (2 for
    uint32_t matrix1_start_offset;    ///< MRAM offset for first matrix
    uint32_t matrix2_start_offset;    ///< MRAM offset for second matrix
    uint32_t result_start_offset;     ///< MRAM offset for result matrix
    uint32_t mem_frame_end;           ///< MRAM offset for end of memory frame
    bool result_valid;              ///< Flag indicating if result is valid
    struct dpu_set_t dpu_set; ///< DPU set for execution
} pim_matrix_multiplication_frame_t;

pim_matrix_multiplication_frame_t* create_pim_matrix_multiplication_frame(uint32_t num_dpus, uint32_t dpu_offset, 
                                                                        uint32_t matrix1_rows, uint32_t matrix1_cols,
                                                                        uint32_t matrix2_rows, uint32_t matrix2_cols,
                                                                        uint32_t result_rows, uint32_t result_cols,
                                                                        uint32_t matrix1_type_size, uint32_t matrix2_type_size, uint32_t result_type_size);

void destroy_pim_matrix_multiplication_frame(pim_matrix_multiplication_frame_t* frame);

void pim_matrix_multiplication_frame_load_first_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix);

void pim_matrix_multiplication_frame_load_second_matrix(pim_matrix_multiplication_frame_t* frame, Matrix * matrix);

void pim_matrix_multiplication_frame_execute(pim_matrix_multiplication_frame_t* frame);

Matrix * pim_matrix_multiplication_frame_get_result(pim_matrix_multiplication_frame_t* frame);

#endif // __PIM_MATRIX_MULTIPLICATION_FRAME_H___