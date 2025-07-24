/**
 * @file matrix.c
 * @brief Implementation of the Matrix struct and related functions.
 */
#include "matrix.h"
#include <string.h>

Matrix* matrix_create_from_2d_array(int16_t rows, int16_t cols, int8_t **data) {
    if (rows <= 0 || cols <= 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat->data[r * cols + c] = data[r][c];
        }
    }
    return mat;
}

Matrix* matrix_create_from_row_major_array(int16_t rows, int16_t cols, int8_t *data) {
    if (rows <= 0 || cols <= 0 || !data) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (int8_t*)malloc(rows * cols * sizeof(int8_t));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    memcpy(mat->data, data, rows * cols * sizeof(int8_t));
    return mat;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

int8_t* matrix_get_row(const Matrix* mat, int r) {
    if (!mat || r < 0 || r >= mat->rows) return NULL;
    return (int8_t*)&mat->data[r * mat->cols];
}

int8_t* matrix_get_col(const Matrix* mat, int c) {
    if (!mat || c < 0 || c >= mat->cols) return NULL;
    int8_t* col = (int8_t*)malloc(mat->rows * sizeof(int8_t));
    if (!col) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        col[r] = mat->data[r * mat->cols + c];
    }
    return col;
}

Matrix* matrix_clone(const Matrix* mat) {
    if (!mat) return NULL;
    Matrix* copy = (Matrix*)malloc(sizeof(Matrix));
    if (!copy) return NULL;
    copy->rows = mat->rows;
    copy->cols = mat->cols;
    copy->data = (int8_t*)malloc(mat->rows * mat->cols * sizeof(int8_t));
    if (!copy->data) {
        free(copy);
        return NULL;
    }
    memcpy(copy->data, mat->data, mat->rows * mat->cols * sizeof(int8_t));
    return copy;
}

bool matrix_compare(const Matrix* a, const Matrix* b) {
    if (!a || !b) return false;
    if (a->rows != b->rows || a->cols != b->cols) return false;
    return memcmp(a->data, b->data, a->rows * a->cols * sizeof(int8_t)) == 0;
}

char* matrix_sprint(const Matrix* mat) {
    if (!mat) return NULL;
    int bufsize = mat->rows * mat->cols * 5 + mat->rows + 1;
    char* buf = (char*)malloc(bufsize);
    if (!buf) return NULL;
    int pos = 0;
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            pos += snprintf(buf + pos, bufsize - pos, "%d ", mat->data[r * mat->cols + c]);
        }
        pos += snprintf(buf + pos, bufsize - pos, "\n");
    }
    buf[pos] = '\0';
    return buf;
}

void matrix_print(const Matrix* mat) {
    if (!mat) return;
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            printf("%d ", mat->data[r * mat->cols + c]);
        }
        printf("\n");
    }
}

int8_t matrix_get(const Matrix* mat, int r, int c) {
    if (!mat || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return 0;
    return mat->data[r * mat->cols + c];
}

int matrix_set(Matrix* mat, int r, int c, int8_t value) {
    if (!mat || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return -1;
    mat->data[r * mat->cols + c] = value;
    return 0;
}
