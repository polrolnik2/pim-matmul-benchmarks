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

int8_t* matrix_get_data_row_major(const Matrix* mat) {
    if (!mat) return NULL;
    int8_t* data = (int8_t*)malloc(mat->rows * mat->cols * sizeof(int8_t));
    if (!data) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        memcpy(data + r * mat->cols, matrix_get_row(mat, r), mat->cols * sizeof(int8_t));
    }
    return data;
}

int8_t* matrix_get_data_column_major(const Matrix* mat) {
    if (!mat) return NULL;
    int8_t* data = (int8_t*)malloc(mat->rows * mat->cols * sizeof(int8_t));
    if (!data) return NULL;
    for (int c = 0; c < mat->cols; ++c) {
        memcpy(data + c * mat->rows, matrix_get_col(mat, c), mat->rows * sizeof(int8_t));
    }
    return data;
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

Matrix** matrix_split_by_rows(const Matrix* mat, int num_submatrices) {
    if (!mat || num_submatrices <= 0 || mat->rows < num_submatrices || mat->rows % num_submatrices != 0) return NULL;
    Matrix** submatrices = (Matrix**)malloc(num_submatrices * sizeof(Matrix*));
    if (!submatrices) return NULL;
    int rows_per_submatrix = mat->rows / num_submatrices;
    for (int i = 0; i < num_submatrices; ++i) {
        int start_row = i * rows_per_submatrix;
        int end_row = (i == num_submatrices - 1) ? mat->rows : start_row + rows_per_submatrix;
        int sub_rows = end_row - start_row;
        int8_t **sub_data = (int8_t**)malloc(sub_rows * sizeof(int8_t*));
        if (!sub_data) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
        for (int r = 0; r < sub_rows; ++r) {
            sub_data[r] = (int8_t*)malloc(mat->cols * sizeof(int8_t));
            if (!sub_data[r]) {
                for (int j = 0; j < r; ++j) free(sub_data[j]);
                free(sub_data);
                for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
                free(submatrices);
                return NULL;
            }
            memcpy(sub_data[r], matrix_get_row(mat, start_row + r), mat->cols * sizeof(int8_t));
        }
        submatrices[i] = matrix_create_from_2d_array(sub_rows, mat->cols, sub_data);
        if (!submatrices[i]) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
    }
    return submatrices;
}

Matrix** matrix_split_by_cols(const Matrix* mat, int num_submatrices) {
    if (!mat || num_submatrices <= 0 || mat->cols < num_submatrices || mat->cols % num_submatrices != 0) return NULL;
    Matrix** submatrices = (Matrix**)malloc(num_submatrices * sizeof(Matrix*));
    if (!submatrices) return NULL;
    int cols_per_submatrix = mat->cols / num_submatrices;
    for (int i = 0; i < num_submatrices; ++i) {
        int start_col = i * cols_per_submatrix;
        int end_col = (i == num_submatrices - 1) ? mat->cols : start_col + cols_per_submatrix;
        int sub_cols = end_col - start_col;
        int8_t * sub_data_column_major = (int8_t*)malloc(mat->rows * sub_cols * sizeof(int8_t));
        if (!sub_data_column_major) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
        for (int c = 0; c < sub_cols; ++c) {
            memcpy(sub_data_column_major + c * mat->rows, matrix_get_col(mat, start_col + c), mat->rows * sizeof(int8_t));
        }
        submatrices[i] = matrix_create_from_column_major_array(mat->rows, sub_cols, sub_data_column_major);
        if (!submatrices[i]) {
            for (int j = 0; j < i; ++j) matrix_free(submatrices[j]);
            free(submatrices);
            return NULL;
        }
    }
    return submatrices;
}

Matrix* matrix_join_by_rows(Matrix** submatrices, int num_submatrices) {
    if (!submatrices || num_submatrices <= 0) return NULL;
    int total_rows = 0;
    int cols = submatrices[0]->cols;
    for (int i = 0; i < num_submatrices; ++i) {
        if (!submatrices[i] || submatrices[i]->cols != cols) {
            return NULL;
        }
        total_rows += submatrices[i]->rows;
    }

    // Create temporary 2D array for matrix data
    int8_t **temp_data = (int8_t**)malloc(total_rows * sizeof(int8_t*));
    if (!temp_data) {
        for (int i = 0; i < num_submatrices; ++i) matrix_free(submatrices[i]);
        free(submatrices);
        return NULL;
    }
    int current_row = 0;
    for (int i = 0; i < num_submatrices; ++i) {
        for (int r = 0; r < submatrices[i]->rows; ++r) {
            temp_data[current_row] = (int8_t*)malloc(cols * sizeof(int8_t));
            if (!temp_data[current_row]) {
                for (int j = 0; j < current_row; ++j) free(temp_data[j]);
                free(temp_data);
                for (int j = 0; j < num_submatrices; ++j) matrix_free(submatrices[j]);
                free(submatrices);
                return NULL;
            }
            memcpy(temp_data[current_row], submatrices[i]->data[r], cols * sizeof(int8_t));
            current_row++;
        }
    }

    // Create matrix from temporary data
    Matrix* mat = matrix_create_from_2d_array(total_rows, cols, temp_data);
    free(temp_data);
    return mat;
}

Matrix* matrix_join_by_cols(Matrix** submatrices, int num_submatrices) {
    if (!submatrices || num_submatrices <= 0) return NULL;
    int total_cols = 0;
    int rows = submatrices[0]->rows;
    for (int i = 0; i < num_submatrices; ++i) {
        if (!submatrices[i] || submatrices[i]->rows != rows) {
            return NULL;
        }
        total_cols += submatrices[i]->cols;
    }

    // Create temporary 2D array for matrix data
    int8_t *temp_data_column_major = (int8_t*)malloc(rows * total_cols * sizeof(int8_t));
    if (!temp_data_column_major) {
        for (int i = 0; i < num_submatrices; ++i) matrix_free(submatrices[i]);
        free(submatrices);
        return NULL;
    }
    int current_col = 0;
    for (int i = 0; i < num_submatrices; ++i) {
        for (int c = 0; c < submatrices[i]->cols; ++c) {
            memcpy(temp_data_column_major + current_col * rows, matrix_get_col(submatrices[i], c), rows * sizeof(int8_t));
            current_col++;
        }
    }

    // Create matrix from temporary data
    Matrix* mat = matrix_create_from_column_major_array(rows, total_cols, temp_data_column_major);
    free(temp_data_column_major);
    return mat;
}
