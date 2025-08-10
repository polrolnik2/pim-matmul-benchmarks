/**
 * @file matrix.c
 * @brief Implementation of the Matrix struct and related functions.
 */
#include "matrix.h"
#include <string.h>

Matrix* matrix_create_from_2d_array(int16_t rows, int16_t cols, void **data, uint32_t element_size) {
    if (rows <= 0 || cols <= 0 || !data || element_size == 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->element_size = element_size;
    mat->data = (void**)malloc(rows * sizeof(void*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = malloc(cols * element_size);
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        memcpy(mat->data[r], data[r], cols * element_size);
    }
    return mat;
}

Matrix* matrix_create_from_row_major_array(int16_t rows, int16_t cols, void *data, uint32_t element_size) {
    if (rows <= 0 || cols <= 0 || !data || element_size == 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->element_size = element_size;
    mat->data = (void**)malloc(rows * sizeof(void*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = malloc(cols * element_size);
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        memcpy(mat->data[r], (char*)data + r * cols * element_size, cols * element_size);
    }
    return mat;
}

Matrix* matrix_create_from_column_major_array(int16_t rows, int16_t cols, void *data, uint32_t element_size) {
    if (rows <= 0 || cols <= 0 || !data || element_size == 0) return NULL;
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->element_size = element_size;
    mat->data = (void**)malloc(rows * sizeof(void*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int r = 0; r < rows; ++r) {
        mat->data[r] = malloc(cols * element_size);
        if (!mat->data[r]) {
            for (int i = 0; i < r; ++i) free(mat->data[i]);
            free(mat->data);
            free(mat);
            return NULL;
        }
        for (int c = 0; c < cols; ++c) {
            memcpy((char*)mat->data[r] + c * element_size, 
                   (char*)data + (c * rows + r) * element_size, 
                   element_size);
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

void* matrix_get_row(const Matrix* mat, int r) {
    if (!mat || r < 0 || r >= mat->rows) return NULL;
    return mat->data[r];
}

void* matrix_get_col(const Matrix* mat, int c) {
    if (!mat || c < 0 || c >= mat->cols) return NULL;
    void* col = malloc(mat->rows * mat->element_size);
    if (!col) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        memcpy((char*)col + r * mat->element_size, 
               (char*)mat->data[r] + c * mat->element_size, 
               mat->element_size);
    }
    return col;
}

void* matrix_get_data_row_major(const Matrix* mat) {
    if (!mat) return NULL;
    void* data = malloc(mat->rows * mat->cols * mat->element_size);
    if (!data) return NULL;
    for (int r = 0; r < mat->rows; ++r) {
        memcpy((char*)data + r * mat->cols * mat->element_size, 
               matrix_get_row(mat, r), 
               mat->cols * mat->element_size);
    }
    return data;
}

void* matrix_get_data_column_major(const Matrix* mat) {
    if (!mat) return NULL;
    void* data = malloc(mat->rows * mat->cols * mat->element_size);
    if (!data) return NULL;
    for (int c = 0; c < mat->cols; ++c) {
        void* col_data = matrix_get_col(mat, c);
        if (!col_data) {
            free(data);
            return NULL;
        }
        memcpy((char*)data + c * mat->rows * mat->element_size, 
               col_data, 
               mat->rows * mat->element_size);
        free(col_data);
    }
    return data;
}

Matrix* matrix_clone(const Matrix* mat) {
    if (!mat) return NULL;
    Matrix* copy = (Matrix*)malloc(sizeof(Matrix));
    if (!copy) return NULL;
    copy->rows = mat->rows;
    copy->cols = mat->cols;
    copy->element_size = mat->element_size;
    copy->data = (void**)malloc(mat->rows * sizeof(void*));
    if (!copy->data) {
        free(copy);
        return NULL;
    }
    for (int r = 0; r < mat->rows; ++r) {
        copy->data[r] = malloc(mat->cols * mat->element_size);
        if (!copy->data[r]) {
            for (int i = 0; i < r; ++i) free(copy->data[i]);
            free(copy->data);
            free(copy);
            return NULL;
        }
        memcpy(copy->data[r], mat->data[r], mat->cols * mat->element_size);
    }
    return copy;
}

bool matrix_compare(const Matrix* a, const Matrix* b) {
    if (!a || !b) return false;
    if (a->rows != b->rows || a->cols != b->cols || a->element_size != b->element_size) return false;
    for (int r = 0; r < a->rows; ++r) {
        if (memcmp(a->data[r], b->data[r], a->cols * a->element_size) != 0) return false;
    }
    return true;
}

char* matrix_sprint(const Matrix* mat, const char* format) {
    if (!mat || !format) return NULL;
    
    // Estimate buffer size (rough approximation)
    int max_element_str_len = 20; // Should be enough for most numeric types
    int bufsize = mat->rows * mat->cols * max_element_str_len + mat->rows + 1;
    char* buf = (char*)malloc(bufsize);
    if (!buf) return NULL;
    
    int pos = 0;
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            // Format based on element size
            if (mat->element_size == sizeof(int8_t)) {
                int8_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(int16_t)) {
                int16_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(int32_t)) {
                int32_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(float)) {
                float value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else if (mat->element_size == sizeof(double)) {
                double value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                pos += snprintf(buf + pos, bufsize - pos, format, value);
            } else {
                // For unknown types, format as hex
                pos += snprintf(buf + pos, bufsize - pos, "0x");
                for (uint32_t k = 0; k < mat->element_size; k++) {
                    pos += snprintf(buf + pos, bufsize - pos, "%02x", 
                                  ((unsigned char*)mat->data[r])[c * mat->element_size + k]);
                }
                pos += snprintf(buf + pos, bufsize - pos, " ");
            }
        }
        pos += snprintf(buf + pos, bufsize - pos, "\n");
    }
    buf[pos] = '\0';
    return buf;
}

void matrix_print(const Matrix* mat, const char* format) {
    if (!mat || !format) return;
    
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            // Print based on element size and format
            if (mat->element_size == sizeof(int8_t)) {
                int8_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(int16_t)) {
                int16_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(int32_t)) {
                int32_t value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(float)) {
                float value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else if (mat->element_size == sizeof(double)) {
                double value;
                memcpy(&value, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
                printf(format, value);
            } else {
                // For unknown types, print as hex bytes
                printf("0x");
                for (uint32_t k = 0; k < mat->element_size; k++) {
                    printf("%02x", ((unsigned char*)mat->data[r])[c * mat->element_size + k]);
                }
                printf(" ");
            }
        }
        printf("\n");
    }
}

int matrix_get(const Matrix* mat, int r, int c, void* out) {
    if (!mat || !out || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return -1;
    memcpy(out, (char*)mat->data[r] + c * mat->element_size, mat->element_size);
    return 0;
}

int matrix_set(Matrix* mat, int r, int c, const void* value) {
    if (!mat || !value || r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) return -1;
    memcpy((char*)mat->data[r] + c * mat->element_size, value, mat->element_size);
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
        submatrices[i] = matrix_create_from_2d_array(sub_rows, mat->cols, (void**)sub_data, mat->element_size);
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
        submatrices[i] = matrix_create_from_column_major_array(mat->rows, sub_cols, sub_data_column_major, mat->element_size);
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

    if (num_submatrices == 1) {
        Matrix* single_matrix = matrix_clone(submatrices[0]);
        return single_matrix;
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
    Matrix* mat = matrix_create_from_2d_array(total_rows, cols, (void**)temp_data, submatrices[0]->element_size);
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
    
    if (num_submatrices == 1) {
        Matrix* single_matrix = matrix_clone(submatrices[0]);
        return single_matrix;
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
    Matrix* mat = matrix_create_from_column_major_array(rows, total_cols, temp_data_column_major, submatrices[0]->element_size);
    free(temp_data_column_major);
    return mat;
}
