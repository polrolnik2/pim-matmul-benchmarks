#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "test_assertions.h"

int test_matrix_create_from_2d_array_and_free() {
    printf("Running test_matrix_create_from_2d_array_and_free...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    ASSERT_EQ(matrix_get(m, 0, 0), 1, "Matrix get (0,0)");
    ASSERT_EQ(matrix_get(m, 0, 1), 2, "Matrix get (0,1)");
    ASSERT_EQ(matrix_get(m, 1, 0), 3, "Matrix get (1,0)");
    ASSERT_EQ(matrix_get(m, 1, 1), 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int test_matrix_create_from_row_major_array_and_free() {
    printf("Running test_matrix_create_from_row_major_array_and_free...\n");
    int8_t data[] = {1, 2, 3, 4};
    Matrix* m = matrix_create_from_row_major_array(2, 2, data);
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    ASSERT_EQ(matrix_get(m, 0, 0), 1, "Matrix get (0,0)");
    ASSERT_EQ(matrix_get(m, 0, 1), 2, "Matrix get (0,1)");
    ASSERT_EQ(matrix_get(m, 1, 0), 3, "Matrix get (1,0)");
    ASSERT_EQ(matrix_get(m, 1, 1), 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int test_matrix_create_from_column_major_array_and_free() {
    printf("Running test_matrix_create_from_column_major_array_and_free...\n");
    int8_t data[] = {1, 3, 2, 4};
    Matrix* m = matrix_create_from_column_major_array(2, 2, data);
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    ASSERT_EQ(matrix_get(m, 0, 0), 1, "Matrix get (0,0)");
    ASSERT_EQ(matrix_get(m, 0, 1), 2, "Matrix get (0,1)");
    ASSERT_EQ(matrix_get(m, 1, 0), 3, "Matrix get (1,0)");
    ASSERT_EQ(matrix_get(m, 1, 1), 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int test_matrix_creation_data_too_small_error() {
    printf("Running test_matrix_creation_data_too_small_error...\n");
    int8_t data[] = {1, 2}; // Only enough for 1 row of 2 cols
    Matrix* m = matrix_create_from_row_major_array(2, 2, data);
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    m = matrix_create_from_column_major_array(2, 2, data);
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    m = matrix_create_from_2d_array(2, 2, (int8_t*[]){data});
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    return 0;
}

int test_matrix_clone_and_compare() {
    printf("Running test_matrix_clone_and_compare...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create_from_2d_array(2, 2, data);
    Matrix* m2 = matrix_clone(m1);
    ASSERT_TRUE(matrix_compare(m1, m2), "Matrix clone/compare failed");
    matrix_set(m2, 0, 0, 9);
    ASSERT_TRUE(!matrix_compare(m1, m2), "Matrix compare after change");
    matrix_free(m1);
    matrix_free(m2);
    return 0;
}

int test_matrix_get_row_col() {
    printf("Running test_matrix_get_row_col...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    int8_t* row = matrix_get_row(m, 1);
    ASSERT_EQ(row[0], 3, "Get row 1 col 0");
    ASSERT_EQ(row[1], 4, "Get row 1 col 1");
    int8_t* row0ptr = matrix_get_row(m, 0);
    ASSERT_EQ(row0ptr[0], 1, "Get row 0 col 0");
    ASSERT_EQ(row0ptr[1], 2, "Get row 0 col 1");
    int8_t* col0 = matrix_get_col(m, 0);
    ASSERT_EQ(col0[0], 1, "Get col 0 row 0");
    ASSERT_EQ(col0[1], 3, "Get col 0 row 1");
    int8_t* col1 = matrix_get_col(m, 1);
    ASSERT_EQ(col1[0], 2, "Get col 1 row 0");
    ASSERT_EQ(col1[1], 4, "Get col 1 row 1");
    free(col0);
    free(col1);
    matrix_free(m);
    return 0;
}

int test_get_row_col_out_of_bounds() {
    printf("Running test_get_row_col_out_of_bounds...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    ASSERT_TRUE(matrix_get_row(m, -1) == NULL, "Get row -1 should fail");
    ASSERT_TRUE(matrix_get_row(m, 2) == NULL, "Get row 2 should fail");
    ASSERT_TRUE(matrix_get_col(m, -1) == NULL, "Get col -1 should fail");
    ASSERT_TRUE(matrix_get_col(m, 2) == NULL, "Get col 2 should fail");
    matrix_free(m);
    return 0;
}

int test_matrix_print_and_sprint() {
    printf("Running test_matrix_print_and_sprint...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    char* str = matrix_sprint(m);
    ASSERT_TRUE(strstr(str, "1 2") && strstr(str, "3 4"), "Matrix sprint");
    free(str);
    matrix_free(m);
    return 0;
}

int test_matrix_row_split_join() {
    printf("Running test_matrix_row_split_join...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* row_1st = matrix_create_from_2d_array(1, 2, (int8_t*[]){row0});
    Matrix* row_2nd = matrix_create_from_2d_array(1, 2, (int8_t*[]){row1});
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    Matrix** submatrices = matrix_split_by_rows(m, 2);
    ASSERT_TRUE(submatrices != NULL && submatrices[0] != NULL && submatrices[1] != NULL, "Row split failed");
    Matrix* joined = matrix_join_by_rows(submatrices, 2);
    ASSERT_TRUE(matrix_compare(m, joined), "Row join failed");
    ASSERT_TRUE(matrix_compare(submatrices[0], row_1st), "First submatrix should match");
    ASSERT_TRUE(matrix_compare(submatrices[1], row_2nd), "Second submatrix should match");
    matrix_free(row_1st);
    matrix_free(row_2nd);
    matrix_free(m);
    matrix_free(submatrices[0]);
    matrix_free(submatrices[1]);
    free(submatrices);
    matrix_free(joined);
    return 0;
}

int test_matrix_col_split_join() {
    printf("Running test_matrix_col_split_join...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* col_1st = matrix_create_from_2d_array(2, 1, (int8_t*[]){(int8_t[]){1}, (int8_t[]){3}});
    Matrix* col_2nd = matrix_create_from_2d_array(2, 1, (int8_t*[]){(int8_t[]){2}, (int8_t[]){4}});
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    Matrix** submatrices = matrix_split_by_cols(m, 2);
    ASSERT_TRUE(submatrices != NULL && submatrices[0] != NULL && submatrices[1] != NULL, "Column split failed");
    ASSERT_TRUE(matrix_compare(submatrices[0], col_1st), "First submatrix should match");
    ASSERT_TRUE(matrix_compare(submatrices[1], col_2nd), "Second submatrix should match");
    // Join the submatrices back into a single matrix
    Matrix* joined = matrix_join_by_cols(submatrices, 2);
    ASSERT_TRUE(matrix_compare(m, joined), "Column join failed");
    matrix_free(col_1st);
    matrix_free(col_2nd);
    matrix_free(m);
    matrix_free(submatrices[0]);
    matrix_free(submatrices[1]);
    free(submatrices);
    matrix_free(joined);
    return 0;
}

int test_matrix_split_into_0_error() {
    printf("Running test_matrix_split_into_0_error...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    Matrix** submatrices = matrix_split_by_rows(m, 0);
    ASSERT_TRUE(submatrices == NULL, "Split into 0 submatrices should fail");
    submatrices = matrix_split_by_cols(m, 0);
    ASSERT_TRUE(submatrices == NULL, "Split into 0 submatrices should fail");
    matrix_free(m);
    return 0;
}

int test_matrix_split_into_more_than_rows_cols_error() {
    printf("Running test_matrix_split_into_more_than_rows_cols_error...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, data);
    Matrix** submatrices = matrix_split_by_rows(m, 3);
    ASSERT_TRUE(submatrices == NULL, "Split into more submatrices than rows should fail");
    submatrices = matrix_split_by_cols(m, 3);
    ASSERT_TRUE(submatrices == NULL, "Split into more submatrices than cols should fail");
    matrix_free(m);
    return 0;
}

int test_split_matrix_indivisible_error() {
    printf("Running test_split_matrix_indivisible_error...\n");
    int8_t row0[] = {1, 2, 3};
    int8_t row1[] = {4, 5, 6};
    int8_t row2[] = {7, 8, 9};
    int8_t* data[] = {row0, row1, row2};
    Matrix* m = matrix_create_from_2d_array(3, 3, data);
    Matrix** submatrices = matrix_split_by_rows(m, 2);
    ASSERT_TRUE(submatrices == NULL, "Split into more submatrices than rows should fail");
    submatrices = matrix_split_by_cols(m, 2);
    ASSERT_TRUE(submatrices == NULL, "Split into more submatrices than cols should fail");
    matrix_free(m);
    return 0;
}

int test_join_unmatching_dimensions_error() {
    printf("Running test_join_unmatching_dimensions_error...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create_from_2d_array(2, 2, data);
    int8_t row0_2[] = {5, 6, 7};
    int8_t row1_2[] = {8, 9, 10};
    int8_t row2_2[] = {11, 12, 13};
    int8_t* data2[] = {row0_2, row1_2, row2_2};
    Matrix* m2 = matrix_create_from_2d_array(3, 3, data2);
    Matrix* joined = matrix_join_by_rows((Matrix*[]){m1, m2}, 2);
    ASSERT_TRUE(joined == NULL, "Joining matrices with unmatching dimensions should fail");
    Matrix* joined_col = matrix_join_by_cols((Matrix*[]){m1, m2}, 2);
    ASSERT_TRUE(joined_col == NULL, "Joining matrices with unmatching dimensions should fail");
    matrix_free(m1);
    matrix_free(m2);
    return 0;
}

int main() {
    int fails = 0;
    fails += test_matrix_create_from_2d_array_and_free();
    fails += test_matrix_create_from_row_major_array_and_free();
    fails += test_matrix_create_from_column_major_array_and_free();
    fails += test_matrix_clone_and_compare();
    fails += test_matrix_get_row_col();
    fails += test_matrix_print_and_sprint();
    fails += test_matrix_row_split_join();
    fails += test_matrix_col_split_join();
    fails += test_matrix_split_into_0_error();
    fails += test_matrix_split_into_more_than_rows_cols_error();
    fails += test_split_matrix_indivisible_error();
    fails += test_join_unmatching_dimensions_error();
    if (fails == 0) {
        printf("[PASS] All matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d matrix tests failed.\n", fails);
        return 1;
    }
}
