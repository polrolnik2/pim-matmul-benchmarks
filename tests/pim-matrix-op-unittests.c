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

#include "matrix.h"
#include "pim_matrix_handle.h"

int test_pim_broadcast_gather() {
    printf("Running test_pim_broadcast_gather...\n");
    simplepim_management_t* table_management = table_management_init(5);
    Matrix* mat = matrix_create_from_2d_array(2, 2, (int8_t*[]) {
        (int8_t[]){1, 2},
        (int8_t[]){3, 4}
    });
    ASSERT_TRUE(mat != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle = broadcast_matrix_to_pim(mat, table_management);
    ASSERT_TRUE(handle != NULL, "Broadcast to PIM failed");
    Matrix* gathered = gather_matrix_from_pim(handle, mat->rows, mat->cols, table_management);
    ASSERT_TRUE(gathered != NULL, "Gather from PIM failed");
    ASSERT_TRUE(matrix_compare(gathered, mat), "Gathered matrix should match original");
    matrix_free(mat);
    matrix_free(gathered);
    free_pim_matrix_handle(handle, table_management);
    return 0;
}

int test_pim_broadcast_free() {
    printf("Running test_pim_broadcast_free...\n");
    simplepim_management_t* table_management = table_management_init(5);
    Matrix* mat = matrix_create_from_2d_array(2, 2, (int8_t*[]) {
        (int8_t[]){1, 2},
        (int8_t[]){3, 4}
    });
    ASSERT_TRUE(mat != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle = broadcast_matrix_to_pim(mat, table_management);
    ASSERT_TRUE(handle != NULL, "Broadcast to PIM failed");
    free_pim_matrix_handle(handle, table_management);
    Matrix* gathered = gather_matrix_from_pim(handle, mat->rows, mat->cols, table_management);
    ASSERT_TRUE(gathered == NULL, "Gather from PIM should fail after free");
    matrix_free(mat);
    return 0;
}

int test_pim_scatter_gather() {
    printf("Running test_pim_scatter_gather...\n");
    simplepim_management_t* table_management = table_management_init(5);
    Matrix* mat = matrix_create_from_2d_array(4, 4, (int8_t*[]) {
        (int8_t[]){1, 2, 5, 6},
        (int8_t[]){3, 4, 7, 8},
        (int8_t[]){9, 10, 13, 14},
        (int8_t[]){11, 12, 15, 16}
    });
    ASSERT_TRUE(mat != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle = scatter_matrix_to_pim(mat, 2, 2, table_management);
    ASSERT_TRUE(handle != NULL, "Scatter to PIM failed");
    Matrix* gathered = gather_matrix_from_pim(handle, mat->rows, mat->cols, table_management);
    ASSERT_TRUE(gathered != NULL, "Gather from PIM failed");
    ASSERT_TRUE(matrix_compare(gathered, mat), "Gathered matrix should match original");
    matrix_free(mat);
    matrix_free(gathered);
    free_pim_matrix_handle(handle, table_management);
    return 0;
}

int test_pim_add_matrices() {
    Matrix* mat1 = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){1, 2},
        (int8_t[]){3, 4}
    });
    Matrix* mat2 = matrix_create_from_2d_array(2, 2, (int8_t*[]) {
        (int8_t[]){5, 6},
        (int8_t[]){7, 8}
    });
    ASSERT_TRUE(mat1 != NULL && mat2 != NULL, "Matrix creation failed");
    
    pim_matrix_handle_t* handle1 = broadcast_matrix_to_pim(mat1, table_management);
    pim_matrix_handle_t* handle2 = broadcast_matrix_to_pim(mat2, table_management);
    pim_matrix_handle_t* handle3 = scatter_matrix_to_pim(mat1, 1, 1, table_management);
    pim_matrix_handle_t* handle4 = scatter_matrix_to_pim(mat2, 1, 1, table_management);
    ASSERT_TRUE(handle1 != NULL && handle2 != NULL && handle3 != NULL && handle4 != NULL, "Broadcast/Scatter to PIM failed");

    Matrix* result1 = gather_matrix_from_pim(handle1, mat1->rows, mat1->cols, table_management);
    ASSERT_TRUE(result1 != NULL, "Gather from PIM failed");
    Matrix* result2 = gather_matrix_from_pim(handle2, mat2->rows, mat2->cols, table_management);
    ASSERT_TRUE(result2 != NULL, "Gather from PIM failed"); 
    Matrix* result3 = gather_matrix_from_pim(handle3, mat1->rows, mat1->cols, table_management);
    ASSERT_TRUE(result3 != NULL, "Gather from PIM failed");
    Matrix* result4 = gather_matrix_from_pim(handle4, mat2->rows, mat2->cols, table_management);
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

int test_adding_scattered_matrices() {
    Matrix* mat1 = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){1, 2},
        (int8_t[]){3, 4}
    });
    Matrix* mat2 = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){5, 6},
        (int8_t[]){7, 8}
    });
    ASSERT_TRUE(mat1 != NULL && mat2 != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle1 = scatter_matrix_to_pim(mat1, 1, 1);
    pim_matrix_handle_t* handle2 = scatter_matrix_to_pim(mat2, 1, 1);
    ASSERT_TRUE(handle1 != NULL && handle2 != NULL, "Scatter to PIM failed");
    pim_matrix_handle_t* result_handle = add_pim_matrices(handle1, handle2);
    ASSERT_TRUE(result_handle != NULL, "Add PIM matrices failed");
    Matrix* result = gather_matrix_from_pim(result_handle->pim_handle, mat1->rows, mat1->cols);
    ASSERT_TRUE(result != NULL, "Gather from PIM failed");
    Matrix* expected = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){6, 8},
        (int8_t[]){10, 12}
    });
    ASSERT_TRUE(matrix_compare(result, expected), "Result matrix should match expected");
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    free_pim_matrix_handle(handle1);
    free_pim_matrix_handle(handle2);
    free_pim_matrix_handle(result_handle);  
    return 0;
}

int test_multiplying_scattered_matrices() {
    Matrix* mat1 = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){1, 2},
        (int8_t[]){3, 4}
    });
    Matrix* mat2 = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){5, 6},
        (int8_t[]){7, 8}
    });
    ASSERT_TRUE(mat1 != NULL && mat2 != NULL, "Matrix creation failed");
    pim_matrix_handle_t* handle1 = scatter_matrix_to_pim(mat1, 1, 1);
    pim_matrix_handle_t* handle2 = scatter_matrix_to_pim(mat2, 1, 1);
    ASSERT_TRUE(handle1 != NULL && handle2 != NULL, "Scatter to PIM failed");
    pim_matrix_handle_t* result_handle = multiply_pim_matrices(handle1, handle2);
    ASSERT_TRUE(result_handle != NULL, "Multiply PIM matrices failed");
    Matrix* result = gather_matrix_from_pim(result_handle->pim_handle, mat1->rows, mat1->cols);
    ASSERT_TRUE(result != NULL, "Gather from PIM failed");
    Matrix* expected = matrix_create(2, 2, (int8_t*[]) {
        (int8_t[]){19, 22},
        (int8_t[]){43, 50}
    });
    ASSERT_TRUE(matrix_compare(result, expected), "Result matrix should match expected");
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    free_pim_matrix_handle(handle1);
    free_pim_matrix_handle(handle2);
    free_pim_matrix_handle(result_handle);
    return 0;
}

int main() {
    int fails = 0;
    fails += test_pim_broadcast_gather();
    fails += test_pim_broadcast_free();
    fails += test_pim_scatter_gather();
    fails += test_pim_add_matrices();
    fails += test_pim_multiply_matrices();
    if (fails == 0) {
        printf("[PASS] All PIM matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d PIM matrix tests failed.\n", fails);
        return 1;
    }
}