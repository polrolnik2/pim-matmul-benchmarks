#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/matrix.h"

// Simple assert macro for C
#define ASSERT_EQ(expr, expected, msg) \
    if ((expr) != (expected)) { \
        printf("[FAIL] %s (expected %d, got %d)\n", msg, (int)(expected), (int)(expr)); \
        return 1; \
    }
#define ASSERT_TRUE(expr, msg) \
    if (!(expr)) { \
        printf("[FAIL] %s\n", msg); \
        return 1; \
    }
#define ASSERT_STR_EQ(expr, expected, msg) \
    if (strcmp((expr), (expected)) != 0) { \
        printf("[FAIL] %s (expected %s, got %s)\n", msg, expected, expr); \
        return 1; \
    }

int test_matrix_create_and_free() {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create(2, 2, data);
    ASSERT_TRUE(m != NULL, "Matrix creation failed");
    ASSERT_EQ(m->rows, 2, "Matrix rows");
    ASSERT_EQ(m->cols, 2, "Matrix cols");
    // Exhaustive element checks
    ASSERT_EQ(matrix_get(m, 0, 0), 1, "Matrix get (0,0)");
    ASSERT_EQ(matrix_get(m, 0, 1), 2, "Matrix get (0,1)");
    ASSERT_EQ(matrix_get(m, 1, 0), 3, "Matrix get (1,0)");
    ASSERT_EQ(matrix_get(m, 1, 1), 4, "Matrix get (1,1)");
    matrix_free(m);
    return 0;
}

int test_matrix_clone_and_compare() {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m1 = matrix_create(2, 2, data);
    Matrix* m2 = matrix_clone(m1);
    ASSERT_TRUE(matrix_compare(m1, m2), "Matrix clone/compare failed");
    matrix_set(m2, 0, 0, 9);
    ASSERT_TRUE(!matrix_compare(m1, m2), "Matrix compare after change");
    matrix_free(m1);
    matrix_free(m2);
    return 0;
}

int test_matrix_get_row_col() {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create(2, 2, data);
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

int test_matrix_print_and_sprint() {
    int8_t row0[] = {1, 2};
    int8_t row1[] = {3, 4};
    int8_t* data[] = {row0, row1};
    Matrix* m = matrix_create(2, 2, data);
    char* str = matrix_sprint(m);
    ASSERT_TRUE(strstr(str, "1 2") && strstr(str, "3 4"), "Matrix sprint");
    free(str);
    matrix_free(m);
    return 0;
}

int main() {
    int fails = 0;
    fails += test_matrix_create_and_free();
    fails += test_matrix_clone_and_compare();
    fails += test_matrix_get_row_col();
    fails += test_matrix_print_and_sprint();
    if (fails == 0) {
        printf("[PASS] All matrix tests passed!\n");
        return 0;
    } else {
        printf("[FAIL] %d matrix tests failed.\n", fails);
        return 1;
    }
}
