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
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    int8_t val;
    matrix_get(m, 0, 0, &val);
    ASSERT_EQ(val, 1, "Matrix get (0,0)");
    matrix_get(m, 0, 1, &val);
    ASSERT_EQ(val, 2, "Matrix get (0,1)");
    matrix_get(m, 1, 0, &val);
    ASSERT_EQ(val, 3, "Matrix get (1,0)");
    matrix_get(m, 1, 1, &val);
    ASSERT_EQ(val, 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = (int8_t*)matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = (int8_t*)matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int test_matrix_create_from_row_major_array_and_free() {
    printf("Running test_matrix_create_from_row_major_array_and_free...\n");
    int8_t data[] = {1, 2, 3, 4};
    Matrix* m = matrix_create_from_row_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    int8_t val;
    matrix_get(m, 0, 0, &val);
    ASSERT_EQ(val, 1, "Matrix get (0,0)");
    matrix_get(m, 0, 1, &val);
    ASSERT_EQ(val, 2, "Matrix get (0,1)");
    matrix_get(m, 1, 0, &val);
    ASSERT_EQ(val, 3, "Matrix get (1,0)");
    matrix_get(m, 1, 1, &val);
    ASSERT_EQ(val, 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = (int8_t*)matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = (int8_t*)matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int test_matrix_create_from_column_major_array_and_free() {
    printf("Running test_matrix_create_from_column_major_array_and_free...\n");
    int8_t data[] = {1, 3, 2, 4};
    Matrix* m = matrix_create_from_column_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    int8_t val;
    matrix_get(m, 0, 0, &val);
    ASSERT_EQ(val, 1, "Matrix get (0,0)");
    matrix_get(m, 0, 1, &val);
    ASSERT_EQ(val, 2, "Matrix get (0,1)");
    matrix_get(m, 1, 0, &val);
    ASSERT_EQ(val, 3, "Matrix get (1,0)");
    matrix_get(m, 1, 1, &val);
    ASSERT_EQ(val, 4, "Matrix get (1,1)");
    int8_t expected_row_major[] = {1, 2, 3, 4};
    int8_t* actual_row_major = (int8_t*)matrix_get_data_row_major(m);
    ASSERT_TRUE(memcmp(expected_row_major, actual_row_major, 4 * sizeof(int8_t)) == 0, "Row major data mismatch");
    free(actual_row_major);
    int8_t expected_col_major[] = {1, 3, 2, 4};
    int8_t* actual_col_major = (int8_t*)matrix_get_data_column_major(m);
    ASSERT_TRUE(memcmp(expected_col_major, actual_col_major, 4 * sizeof(int8_t)) == 0, "Column major data mismatch");
    free(actual_col_major);
    matrix_free(m);
    return 0;
}

int test_matrix_creation_data_too_small_error() {
    printf("Running test_matrix_creation_data_too_small_error...\n");
    int8_t data[] = {1, 2}; // Only enough for 1 row of 2 cols
    Matrix* m = matrix_create_from_row_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    m = matrix_create_from_column_major_array(2, 2, data, sizeof(int8_t));
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    m = matrix_create_from_2d_array(2, 2, (void*[]){data}, sizeof(int8_t));
    ASSERT_TRUE(m == NULL, "Matrix creation should fail with insufficient data");
    return 0;
}

int test_matrix_clone_and_compare() {
    printf("Running test_matrix_clone_and_compare...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    Matrix* m2 = matrix_clone(m1);
    ASSERT_TRUE(matrix_compare(m1, m2), "Matrix clone/compare failed");
    int8_t new_val = 9;
    matrix_set(m2, 0, 0, &new_val);
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
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    int8_t* row = (int8_t*)matrix_get_row(m, 1);
    ASSERT_EQ(row[0], 3, "Get row 1 col 0");
    ASSERT_EQ(row[1], 4, "Get row 1 col 1");
    int8_t* row0ptr = (int8_t*)matrix_get_row(m, 0);
    ASSERT_EQ(row0ptr[0], 1, "Get row 0 col 0");
    ASSERT_EQ(row0ptr[1], 2, "Get row 0 col 1");
    int8_t* col0 = (int8_t*)matrix_get_col(m, 0);
    ASSERT_EQ(col0[0], 1, "Get col 0 row 0");
    ASSERT_EQ(col0[1], 3, "Get col 0 row 1");
    int8_t* col1 = (int8_t*)matrix_get_col(m, 1);
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
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
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
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    char* str = matrix_sprint(m, "%d ");
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
    Matrix* row_1st = matrix_create_from_2d_array(1, 2, (void*[]){row0}, sizeof(int8_t));
    Matrix* row_2nd = matrix_create_from_2d_array(1, 2, (void*[]){row1}, sizeof(int8_t));
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
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
    int8_t col1_data[] = {1, 3};
    int8_t col2_data[] = {2, 4};
    int8_t* col1_rows[] = {&col1_data[0], &col1_data[1]};
    int8_t* col2_rows[] = {&col2_data[0], &col2_data[1]};
    Matrix* col_1st = matrix_create_from_2d_array(2, 1, (void**)col1_rows, sizeof(int8_t));
    Matrix* col_2nd = matrix_create_from_2d_array(2, 1, (void**)col2_rows, sizeof(int8_t));
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
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
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
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
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
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
    Matrix* m = matrix_create_from_2d_array(3, 3, (void**)data, sizeof(int8_t));
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
    Matrix* m1 = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    int8_t row0_2[] = {5, 6, 7};
    int8_t row1_2[] = {8, 9, 10};
    int8_t row2_2[] = {11, 12, 13};
    int8_t* data2[] = {row0_2, row1_2, row2_2};
    Matrix* m2 = matrix_create_from_2d_array(3, 3, (void**)data2, sizeof(int8_t));
    Matrix* joined = matrix_join_by_rows((Matrix*[]){m1, m2}, 2);
    ASSERT_TRUE(joined == NULL, "Joining matrices with unmatching dimensions should fail");
    Matrix* joined_col = matrix_join_by_cols((Matrix*[]){m1, m2}, 2);
    ASSERT_TRUE(joined_col == NULL, "Joining matrices with unmatching dimensions should fail");
    matrix_free(m1);
    matrix_free(m2);
    return 0;
}

int test_matrix_add_rows_with_zero_fill() {
    printf("Running test_matrix_add_rows_with_zero_fill...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 2 rows with default zero fill
    Matrix* extended = matrix_add_rows(m, 2, NULL);
    ASSERT_TRUE(extended != NULL, "Matrix add rows failed");
    ASSERT_EQ(extended->rows, 4, "Extended matrix should have 4 rows");
    ASSERT_EQ(extended->cols, 2, "Extended matrix should have 2 cols");
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    ASSERT_EQ(val, 1, "Original data (0,0)");
    matrix_get(extended, 0, 1, &val);
    ASSERT_EQ(val, 2, "Original data (0,1)");
    matrix_get(extended, 1, 0, &val);
    ASSERT_EQ(val, 3, "Original data (1,0)");
    matrix_get(extended, 1, 1, &val);
    ASSERT_EQ(val, 4, "Original data (1,1)");
    
    // Check new rows are zero-filled
    matrix_get(extended, 2, 0, &val);
    ASSERT_EQ(val, 0, "New row (2,0) should be zero");
    matrix_get(extended, 2, 1, &val);
    ASSERT_EQ(val, 0, "New row (2,1) should be zero");
    matrix_get(extended, 3, 0, &val);
    ASSERT_EQ(val, 0, "New row (3,0) should be zero");
    matrix_get(extended, 3, 1, &val);
    ASSERT_EQ(val, 0, "New row (3,1) should be zero");
    
    matrix_free(m);
    matrix_free(extended);
    return 0;
}

int test_matrix_add_rows_with_custom_fill() {
    printf("Running test_matrix_add_rows_with_custom_fill...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 1 row with custom fill value
    int8_t fill_value = 9;
    Matrix* extended = matrix_add_rows(m, 1, &fill_value);
    ASSERT_TRUE(extended != NULL, "Matrix add rows failed");
    ASSERT_EQ(extended->rows, 3, "Extended matrix should have 3 rows");
    ASSERT_EQ(extended->cols, 2, "Extended matrix should have 2 cols");
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    ASSERT_EQ(val, 1, "Original data (0,0)");
    matrix_get(extended, 1, 1, &val);
    ASSERT_EQ(val, 4, "Original data (1,1)");
    
    // Check new row is filled with custom value
    matrix_get(extended, 2, 0, &val);
    ASSERT_EQ(val, 9, "New row (2,0) should be filled value");
    matrix_get(extended, 2, 1, &val);
    ASSERT_EQ(val, 9, "New row (2,1) should be filled value");
    
    matrix_free(m);
    matrix_free(extended);
    return 0;
}

int test_matrix_add_cols_with_zero_fill() {
    printf("Running test_matrix_add_cols_with_zero_fill...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 2 columns with default zero fill
    Matrix* extended = matrix_add_cols(m, 2, NULL);
    ASSERT_TRUE(extended != NULL, "Matrix add cols failed");
    ASSERT_EQ(extended->rows, 2, "Extended matrix should have 2 rows");
    ASSERT_EQ(extended->cols, 4, "Extended matrix should have 4 cols");
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    ASSERT_EQ(val, 1, "Original data (0,0)");
    matrix_get(extended, 0, 1, &val);
    ASSERT_EQ(val, 2, "Original data (0,1)");
    matrix_get(extended, 1, 0, &val);
    ASSERT_EQ(val, 3, "Original data (1,0)");
    matrix_get(extended, 1, 1, &val);
    ASSERT_EQ(val, 4, "Original data (1,1)");
    
    // Check new columns are zero-filled
    matrix_get(extended, 0, 2, &val);
    ASSERT_EQ(val, 0, "New col (0,2) should be zero");
    matrix_get(extended, 0, 3, &val);
    ASSERT_EQ(val, 0, "New col (0,3) should be zero");
    matrix_get(extended, 1, 2, &val);
    ASSERT_EQ(val, 0, "New col (1,2) should be zero");
    matrix_get(extended, 1, 3, &val);
    ASSERT_EQ(val, 0, "New col (1,3) should be zero");
    
    matrix_free(m);
    matrix_free(extended);
    return 0;
}

int test_matrix_add_cols_with_custom_fill() {
    printf("Running test_matrix_add_cols_with_custom_fill...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Add 1 column with custom fill value
    int8_t fill_value = 7;
    Matrix* extended = matrix_add_cols(m, 1, &fill_value);
    ASSERT_TRUE(extended != NULL, "Matrix add cols failed");
    ASSERT_EQ(extended->rows, 2, "Extended matrix should have 2 rows");
    ASSERT_EQ(extended->cols, 3, "Extended matrix should have 3 cols");
    
    // Check original data is preserved
    int8_t val;
    matrix_get(extended, 0, 0, &val);
    ASSERT_EQ(val, 1, "Original data (0,0)");
    matrix_get(extended, 1, 1, &val);
    ASSERT_EQ(val, 4, "Original data (1,1)");
    
    // Check new column is filled with custom value
    matrix_get(extended, 0, 2, &val);
    ASSERT_EQ(val, 7, "New col (0,2) should be filled value");
    matrix_get(extended, 1, 2, &val);
    ASSERT_EQ(val, 7, "New col (1,2) should be filled value");
    
    matrix_free(m);
    matrix_free(extended);
    return 0;
}

int test_matrix_add_rows_cols_error_cases() {
    printf("Running test_matrix_add_rows_cols_error_cases...\n");
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create_from_2d_array(2, 2, (void**)data, sizeof(int8_t));
    
    // Test error cases
    Matrix* result = matrix_add_rows(NULL, 1, NULL);
    ASSERT_TRUE(result == NULL, "Add rows with NULL matrix should fail");
    
    result = matrix_add_rows(m, 0, NULL);
    ASSERT_TRUE(result == NULL, "Add 0 rows should fail");
    
    result = matrix_add_rows(m, -1, NULL);
    ASSERT_TRUE(result == NULL, "Add negative rows should fail");
    
    result = matrix_add_cols(NULL, 1, NULL);
    ASSERT_TRUE(result == NULL, "Add cols with NULL matrix should fail");
    
    result = matrix_add_cols(m, 0, NULL);
    ASSERT_TRUE(result == NULL, "Add 0 cols should fail");
    
    result = matrix_add_cols(m, -1, NULL);
    ASSERT_TRUE(result == NULL, "Add negative cols should fail");
    
    matrix_free(m);
    return 0;
}

int test_matrix_add_rows_cols_different_types() {
    printf("Running test_matrix_add_rows_cols_different_types...\n");
    
    // Test with int16_t
    int16_t row0_16[] = {100, 200};
    int16_t row1_16[] = {300, 400};
    int16_t* data_16[] = {row0_16, row1_16};
    Matrix* m16 = matrix_create_from_2d_array(2, 2, (void**)data_16, sizeof(int16_t));
    
    int16_t fill_16 = 999;
    Matrix* extended_16 = matrix_add_rows(m16, 1, &fill_16);
    ASSERT_TRUE(extended_16 != NULL, "Matrix add rows int16 failed");
    ASSERT_EQ(extended_16->rows, 3, "Extended int16 matrix should have 3 rows");
    
    int16_t val_16;
    matrix_get(extended_16, 2, 0, &val_16);
    ASSERT_EQ(val_16, 999, "New row should have fill value");
    
    matrix_free(m16);
    matrix_free(extended_16);
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
    fails += test_matrix_add_rows_with_zero_fill();
    fails += test_matrix_add_rows_with_custom_fill();
    fails += test_matrix_add_cols_with_zero_fill();
    fails += test_matrix_add_cols_with_custom_fill();
    fails += test_matrix_add_rows_cols_error_cases();
    fails += test_matrix_add_rows_cols_different_types();
    if (fails == 0) {
        printf("[PASS] All matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d matrix tests failed.\n", fails);
        return 1;
    }
}
