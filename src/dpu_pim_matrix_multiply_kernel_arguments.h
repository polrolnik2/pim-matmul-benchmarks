#ifndef __DPU_PIM_MATRIX_MULTIPLY_KERNEL_ARGUMENTS_H___
#define __DPU_PIM_MATRIX_MULTIPLY_KERNEL_ARGUMENTS_H___

typedef struct {
    uint32_t matrix1_start_offset;
    uint32_t matrix2_start_offset;
    uint32_t result_start_offset;
    uint32_t matrix1_rows;
    uint32_t matrix1_cols;
    uint32_t matrix2_rows;
    uint32_t matrix2_cols;
    uint32_t result_rows;
    uint32_t result_cols;
    uint32_t matrix1_type_size;
    uint32_t matrix2_type_size;
    uint32_t result_type_size;
} dpu_pim_matrix_multiply_kernel_arguments_t;

#endif // __DPU_PIM_MATRIX_MULTIPLY_KERNEL_ARGUMENTS_H___