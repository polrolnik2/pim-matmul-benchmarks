#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_assertions.h"

#include "processing/map/Map.h"
#include "processing/zip/Zip.h"
#include "processing/gen_red/GenRed.h"
#include "processing/ProcessingHelperHost.h"
#include "communication/CommOps.h"
#include "management/Management.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>

#include "matrix.h"
#include "pim_matrix_handle.h"

int test_pim_broadcast_gather(simplepim_management_t* table_management) {
    printf("Running test_pim_broadcast_gather...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* mat = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    ASSERT_TRUE(mat != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle = broadcast_matrix_to_pim(mat, table_management);
    ASSERT_TRUE(handle != NULL, "Broadcast to PIM failed");
    Matrix* gathered = gather_matrix_from_pim(handle, mat->rows, mat->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(gathered != NULL, "Gather from PIM failed");
    ASSERT_TRUE(matrix_compare(gathered, mat), "Gathered matrix should match original");
    matrix_free(mat);
    matrix_free(gathered);
    free_pim_matrix_handle(handle, table_management);
    return 0;
}

int test_pim_broadcast_free(simplepim_management_t* table_management) {
    printf("Running test_pim_broadcast_free...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* mat = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    ASSERT_TRUE(mat != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle = broadcast_matrix_to_pim(mat, table_management);
    ASSERT_TRUE(handle != NULL, "Broadcast to PIM failed");
    free_pim_matrix_handle(handle, table_management);
    Matrix* gathered = gather_matrix_from_pim(handle, mat->rows, mat->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(gathered == NULL, "Gather from PIM should fail after free");
    matrix_free(mat);
    return 0;
}

int test_pim_scatter_gather(simplepim_management_t* table_management) {
    printf("Running test_pim_scatter_gather...\n");
    int8_t row0[] = {1, 2, 5, 6};
    int8_t row1[] = {3, 4, 7, 8};
    int8_t row2[] = {9, 10, 13, 14};
    int8_t row3[] = {11, 12, 15, 16};
    int8_t* data[] = {row0, row1, row2, row3};
    Matrix* mat = matrix_create_from_2d_array(4, 4, (void**)data, sizeof(int8_t));
    ASSERT_TRUE(mat != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle = scatter_matrix_to_pim(mat, 2, 2, table_management);
    ASSERT_TRUE(handle != NULL, "Scatter to PIM failed");
    Matrix* gathered = gather_matrix_from_pim(handle, mat->rows, mat->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(gathered != NULL, "Gather from PIM failed");
    ASSERT_TRUE(matrix_compare(gathered, mat), "Gathered matrix should match original");
    matrix_free(mat);
    matrix_free(gathered);
    free_pim_matrix_handle(handle, table_management);
    return 0;
}

// Test for mantaining data integrity with broadcasting scattering and gathering with multiple PIM matrices
int test_pim_broadcast_multiple_scatter_gather(simplepim_management_t* table_management) {
    printf("Running test_pim_broadcast_multiple_scatter_gather...\n");
    int8_t mat1_row0[] = {1, 2};
    int8_t mat1_row1[] = {3, 4};
    int8_t* mat1_data[] = {mat1_row0, mat1_row1};
    Matrix* mat1 = matrix_create_from_2d_array(2, 2, (void**)mat1_data, sizeof(int8_t));
    
    int8_t mat2_row0[] = {5, 6};
    int8_t mat2_row1[] = {7, 8};
    int8_t* mat2_data[] = {mat2_row0, mat2_row1};
    Matrix* mat2 = matrix_create_from_2d_array(2, 2, (void**)mat2_data, sizeof(int8_t));
    ASSERT_TRUE(mat1 != NULL && mat2 != NULL, "Matrix creation failed");
    
    pim_matrix_handle_t* handle1 = broadcast_matrix_to_pim(mat1, table_management);
    pim_matrix_handle_t* handle2 = broadcast_matrix_to_pim(mat2, table_management);
    pim_matrix_handle_t* handle3 = scatter_matrix_to_pim(mat1, 1, 1, table_management);
    pim_matrix_handle_t* handle4 = scatter_matrix_to_pim(mat2, 1, 1, table_management);
    ASSERT_TRUE(handle1 != NULL && handle2 != NULL && handle3 != NULL && handle4 != NULL, "Broadcast/Scatter to PIM failed");

    Matrix* result1 = gather_matrix_from_pim(handle1, mat1->rows, mat1->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(result1 != NULL, "Gather from PIM failed");
    Matrix* result2 = gather_matrix_from_pim(handle2, mat2->rows, mat2->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(result2 != NULL, "Gather from PIM failed"); 
    Matrix* result3 = gather_matrix_from_pim(handle3, mat1->rows, mat1->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(result3 != NULL, "Gather from PIM failed");
    Matrix* result4 = gather_matrix_from_pim(handle4, mat2->rows, mat2->cols, sizeof(int8_t), table_management);
    ASSERT_TRUE(result4 != NULL, "Gather from PIM failed");
    ASSERT_TRUE(matrix_compare(result1, mat1), "Result matrix 1 should match original");
    ASSERT_TRUE(matrix_compare(result2, mat2), "Result matrix 2 should match original");
    ASSERT_TRUE(matrix_compare(result3, mat1), "Result matrix 3 should match original");
    ASSERT_TRUE(matrix_compare(result4, mat2), "Result matrix 4 should match original");
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result1);
    matrix_free(result2);
    matrix_free(result3);
    matrix_free(result4);
    free_pim_matrix_handle(handle1, table_management);
    free_pim_matrix_handle(handle2, table_management);
    free_pim_matrix_handle(handle3, table_management);
    free_pim_matrix_handle(handle4, table_management);
    return 0;
}

int test_pim_multiply_matrices(simplepim_management_t* table_management) {
    int8_t mat1_row0[] = {1, 2, 3, 4};
    int8_t mat1_row1[] = {5, 6, 7, 8};
    int8_t mat1_row2[] = {9, 10, 11, 12};
    int8_t mat1_row3[] = {13, 14, 15, 16};
    int8_t* mat1_data[] = {mat1_row0, mat1_row1, mat1_row2, mat1_row3};
    Matrix* mat1 = matrix_create_from_2d_array(4, 4, (void**)mat1_data, sizeof(int8_t));

    // Transpose mat2 before multiplication
    int8_t mat2_row0[] = {1, 5, 9, 13};
    int8_t mat2_row1[] = {2, 6, 10, 14};
    int8_t mat2_row2[] = {3, 7, 11, 15};
    int8_t mat2_row3[] = {4, 8, 12, 16};
    int8_t* mat2_data[] = {mat2_row0, mat2_row1, mat2_row2, mat2_row3};
    Matrix* mat2 = matrix_create_from_2d_array(4, 4, (void**)mat2_data, sizeof(int8_t));
    ASSERT_TRUE(mat1 != NULL && mat2 != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle1 = broadcast_matrix_to_pim(mat1, table_management);
    pim_matrix_handle_t* handle2 = broadcast_matrix_to_pim(mat2, table_management);
    ASSERT_TRUE(handle1 != NULL && handle2 != NULL, "Broadcast to PIM failed");
    pim_matrix_handle_t* result_handle = multiply_pim_matrices(handle1, handle2, table_management);
    ASSERT_TRUE(result_handle != NULL, "Multiply PIM matrices failed");
    Matrix* result = gather_matrix_from_pim(result_handle, mat1->rows, mat1->cols, sizeof(int16_t), table_management);
    ASSERT_TRUE(result != NULL, "Gather from PIM failed");
    Matrix* expected = matrix_create_from_2d_array(4, 4, (int16_t*[]) {
        (int16_t[]){90, 100, 110, 120},
        (int16_t[]){202, 228, 254, 280},
        (int16_t[]){314, 356, 398, 440},
        (int16_t[]){426, 484, 542, 600},
    }, sizeof(int16_t));
    ASSERT_TRUE(matrix_compare(result, expected), "Result matrix should match expected");
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    free_pim_matrix_handle(handle1, table_management);
    free_pim_matrix_handle(handle2, table_management);
    free_pim_matrix_handle(result_handle, table_management);
    return 0;
}

int test_multiplying_scattered_matrices(simplepim_management_t* table_management) {
    int8_t mat1_row0[] = {1, 2, 3, 4};
    int8_t mat1_row1[] = {5, 6, 7, 8};
    int8_t mat1_row2[] = {9, 10, 11, 12};
    int8_t mat1_row3[] = {13, 14, 15, 16};
    int8_t* mat1_data[] = {mat1_row0, mat1_row1, mat1_row2, mat1_row3};
    Matrix* mat1 = matrix_create_from_2d_array(4, 4, (void**)mat1_data, sizeof(int8_t));
    // Transpose mat2 before multiplication
    int8_t mat2_row0[] = {1, 5, 9, 13};
    int8_t mat2_row1[] = {2, 6, 10, 14};
    int8_t mat2_row2[] = {3, 7, 11, 15};
    int8_t mat2_row3[] = {4, 8, 12, 16};
    int8_t* mat2_data[] = {mat2_row0, mat2_row1, mat2_row2, mat2_row3};
    Matrix* mat2 = matrix_create_from_2d_array(4, 4, (void**)mat2_data, sizeof(int8_t));
    Matrix* submatrices[] = {mat2, mat2, mat2, mat2}; // Create 4 submatrices of mat2
    mat2 = matrix_join_by_rows(submatrices, 4); // Join to make it 8x4
    ASSERT_TRUE(mat1 != NULL && mat2 != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle1 = broadcast_matrix_to_pim(mat1, table_management);
    pim_matrix_handle_t* handle2 = scatter_matrix_to_pim(mat2, 4, 4, table_management);
    ASSERT_TRUE(handle1 != NULL && handle2 != NULL, "Scatter to PIM failed");
    pim_matrix_handle_t* result_handle = multiply_pim_matrices(handle1, handle2, table_management);
    ASSERT_TRUE(result_handle != NULL, "Multiply PIM matrices failed");
    Matrix* result = gather_matrix_from_pim(result_handle, mat1->rows, mat1->cols, sizeof(int16_t), table_management);
    printf("Result:\n%s\n", matrix_sprint(result, "| %d |"));
    ASSERT_TRUE(result != NULL, "Gather from PIM failed");
    Matrix* expected = matrix_create_from_2d_array(4, 4, (int8_t*[]) {
        (int8_t[]){250, 260, 270, 280},
        (int8_t[]){618, 644, 670, 696},
        (int8_t[]){986, 1028, 1070, 1112},
        (int8_t[]){1354, 1412, 1470, 1528}
    }, sizeof(int8_t));
    ASSERT_TRUE(matrix_compare(result, expected), "Result matrix should match expected");
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    free_pim_matrix_handle(handle1, table_management);
    free_pim_matrix_handle(handle2, table_management);
    free_pim_matrix_handle(result_handle, table_management);
    return 0;
}

int main() {
    int fails = 0;
    simplepim_management_t* table_management = table_management_init(4);
    fails += test_pim_broadcast_gather(table_management);
    fails += test_pim_broadcast_free(table_management);
    fails += test_pim_scatter_gather(table_management);
    fails += test_pim_broadcast_multiple_scatter_gather(table_management);
    fails += test_multiplying_scattered_matrices(table_management);
    fails += test_pim_multiply_matrices(table_management);
    fails += test_multiplying_scattered_matrices(table_management);
    if (fails == 0) {
        printf("[PASS] All PIM matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d PIM matrix tests failed.\n", fails);
        return 1;
    }
}