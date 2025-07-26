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
    mat->data = (int8_t**)malloc(rows * sizeof(int8_t*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = (int8_t*)malloc(cols * sizeof(int8_t));
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        memcpy(mat->data[r], data[r], cols * sizeof(int8_t));
    }
    return mat;
}

Matrix* matrix_create_from_row_major_array(int16_t rows, int16_t cols, int8_t *data) {
    if (rows <= 0 || cols <= 0 || !data) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (int8_t**)malloc(rows * sizeof(int8_t*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = (int8_t*)malloc(cols * sizeof(int8_t));
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        memcpy(mat->data[r], data + r * cols, cols * sizeof(int8_t));
    }
    return mat;
}

Matrix* matrix_create_from_column_major_array(int16_t rows, int16_t cols, int8_t *data) {
    if (rows <= 0 || cols <= 0 || !data) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (int8_t**)malloc(rows * sizeof(int8_t*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = (int8_t*)malloc(cols * sizeof(int8_t));
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        for (int c = 0; c < cols; ++c) {
            mat->data[r][c] = data[c * rows + r];
        }
    }
    return mat;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        if (mat->data) {
            for (int r = 0; r < mat->rows; ++r) {
                free(mat->data[r]);
            }
            free(mat->data);
        }
        free(mat);
    }
}

int8_t* matrix_get_row(const Matrix* mat, int r) {
    if (!mat || r < 0 || r >= mat->rows) return NULL;
    return mat->data[r];
}

int8_t* matrix_get_col(const Matrix* mat, int c) {
    if (!mat || c < 0 || c >= mat->cols) return NULL;
    int8_t* col = (int8_t*)malloc(mat->rows * sizeof(int8_t));
    if (!col) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        col[r] = mat->data[r][c];
    }
    return col;
}

Matrix* matrix_clone(const Matrix* mat) {
    if (!mat) return NULL;
    Matrix* copy = (Matrix*)malloc(sizeof(Matrix));
    if (!copy) return NULL;
    copy->rows = mat->rows;
    copy->cols = mat->cols;
    copy->data = (int8_t**)malloc(mat->rows * sizeof(int8_t*));
    if (!copy->data) {
        free(copy);
        return NULL;
    }
    for (int r = 0; r < mat->rows; ++r) {
        copy->data[r] = (int8_t*)malloc(mat->cols * sizeof(int8_t));
        if (!copy->data[r]) {
            for (int i = 0; i < r; ++i) free(copy->data[i]);
            free(copy->data);
            free(copy);
            return NULL;
        }
        memcpy(copy->data[r], mat->data[r], mat->cols * sizeof(int8_t));
    }
    return copy;
}

bool matrix_compare(const Matrix* a, const Matrix* b) {
    if (!a || !b) return false;
    if (a->rows != b->rows || a->cols != b->cols) return false;
    for (int r = 0; r < a->rows; ++r) {
        if (memcmp(a->data[r], b->data[r], a->cols * sizeof(int8_t)) != 0) return false;
    }
    return true;
}

char* matrix_sprint(const Matrix* mat) {
    if (!mat) return NULL;
    int bufsize = mat->rows * mat->cols * 5 + mat->rows + 1;
    char* buf = (char*)malloc(bufsize);
    if (!buf) return NULL;
    int pos = 0;
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            pos += snprintf(buf + pos, bufsize - pos, "%d ", mat->data[r][c]);
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
            printf("%d ", mat->data[r][c]);
        }
        printf("\n");
    }
}

int8_t matrix_get(const Matrix* mat, int r, int c) {
    if (!mat || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return 0;
    return mat->data[r][c];
}

int matrix_set(Matrix* mat, int r, int c, int8_t value) {
    if (!mat || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return -1;
    mat->data[r][c] = value;
    return 0;
}
