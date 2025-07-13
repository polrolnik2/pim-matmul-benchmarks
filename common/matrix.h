#ifndef MATRIX_H
#define MATRIX_H

struct Matrix {
    int16_t rows;
    int16_t cols;
    int8_t* data;

    // Constructor to initialize the matrix with given dimensions
    Matrix(int16_t rows, int16_t cols, int8_t **data) : rows(rows), cols(cols) {
        this->data = new int8_t[rows * cols];
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                this->data[r * cols + c] = data[r][c];
            }
        }
    }

    // Function to get a specific row as a pointer
    int8_t* get_row(int r) const {
        return &data[r * cols];
    }

    // Function to get a specific column as a dynamically allocated array
    int8_t* get_col(int c) const {
        int8_t* col = new int8_t[rows];
        for (int r = 0; r < rows; ++r) {
            col[r] = data[r * cols + c];
        }
        return col;
    }
    
    // Clone the matrix (deep copy)
    Matrix* clone() const {
        int8_t** temp = new int8_t*[rows];
        for (int r = 0; r < rows; ++r) {
            temp[r] = new int8_t[cols];
            for (int c = 0; c < cols; ++c) {
                temp[r][c] = data[r * cols + c];
            }
        }
        Matrix* copy = new Matrix(rows, cols, temp);
        for (int r = 0; r < rows; ++r) {
            delete[] temp[r];
        }
        delete[] temp;
        return copy;
    }

    // Compare two matrices for equality
    bool compare(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) return false;
        for (int i = 0; i < rows * cols; ++i) {
            if (data[i] != other.data[i]) return false;
        }
        return true;
    }

    // Print the matrix to stdout
    void sprint() const {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                printf("%d ", data[r * cols + c]);
            }
            printf("\n");
        }
    }

    // Destructor to free allocated memory
    ~Matrix() {
        delete[] data;
    }

    // Function to access elements in the matrix
    double& operator()(int r, int c) {
        return data[r * cols + c];
    }
};

#endif // MATRIX_H