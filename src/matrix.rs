pub struct Matrix {
    pub row: usize,
    pub col: usize,
    pub data: Vec<f32>,
}
impl Matrix {
    pub fn new(row: usize, col: usize, data: Option<Vec<f32>>) -> Matrix {
        let matrix_data = match data {
            Some(d) => {
                if d.len() != (row * col) {
                    eprint!(
                        "Provided data length does not match row * col. Returning same matrix."
                    );
                    vec![0.0; row * col]
                } else {
                    d
                }
            }
            None => vec![0.0; row * col], // Default to 0.0
        };
        Matrix {
            row,
            col,
            data: matrix_data,
        }
    }

    /// Performs the dot product between two matrices. The order is: M1 * M2, with
    /// M1 being self.
    pub fn dot_product_matrix(&self, matrix: Matrix) -> Option<Matrix> {
        // check the matrices can be dot produc'd
        if self.col != matrix.row {
            return None;
        }
        let mut result = Matrix::new(self.row, matrix.col, None);
        // for each row of matrix self
        for r in 0..self.row {
            // for each column of matrix 2
            for c in 0..matrix.col {
                let mut sum = 0.0 as f32;
                // Dot product of row r of self and column c of matrix
                for k in 0..self.col {
                    sum += self.data[r * self.col + k] * matrix.data[k * matrix.col + c];
                }
                result.data[r * matrix.col + c] = sum;
            }
        }
        Some(result)
    }

    ///Performs the dot product between a matrix and a vector. The order is: M * v.
    /// The signature has got a slice since this is more idiomatic and allows the management
    /// of a pure reference to a contiguous data structure with respect referencing Vec<> which
    /// will take also the heap.
    pub fn dot_product_vector(&self, v: &[f32]) -> Option<Vec<f32>> {
        if self.col != v.len() {
            return None;
        }
        let mut output = vec![0.0; self.row];
        for r in 0..self.row {
            let mut sum = 0.0;
            for c in 0..self.col {
                sum += self.data[r * self.col + c] * v[c];
            }
            output[r] = sum;
        }
        Some(output)
    }

    /// Performs the dot product between a transposed matrix and a vector. So M' * v, with ' = Transposed
    pub fn transpose_dot_vector(&self, v: &[f32]) -> Option<Vec<f32>> {
        if self.row != v.len() {
            return None;
        }
        let mut result = vec![0.0; self.col];
        for i in 0..self.col {
            let mut sum = 0.0;
            for j in 0..self.row {
                sum += self.data[j * self.col + i] * v[j];
            }
            result[i] = sum;
        }
        Some(result)
    }

    /// Perform the ReLU on a Matrix. Return a new Matrix.
    pub fn relu(&self) -> Matrix {
        let mut activated = Matrix::new(self.row, self.col, None);
        for i in 0..self.data.len() {
            activated.data[i] = self.data[i].max(0.0);
        }
        activated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix() {
        let m = Matrix::new(2, 3, None);
        assert_eq!(m.row, 2);
        assert_eq!(m.col, 3);
        assert_eq!(m.data.len(), 6);
        assert!(m.data.iter().all(|&x| x == 0.0));
        let m2 = Matrix::new(2, 3, Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        assert_eq!(m2.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Test fallback to zeros when length is incorrect
        let m3 = Matrix::new(2, 3, Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert!(m3.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dot_product_matrix() {
        let a = Matrix {
            row: 2,
            col: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let b = Matrix {
            row: 3,
            col: 2,
            data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        };
        let result = a.dot_product_matrix(b).unwrap();
        assert_eq!(result.row, 2);
        assert_eq!(result.col, 2);
        // result multiplied by hand
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_dot_product_vector() {
        let a = Matrix {
            row: 2,
            col: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let b = vec![1.0, 2.0, 3.0];
        let result = a.dot_product_vector(&b);
        // result multiplied by hand
        assert_eq!(result, Some(vec![14.0, 32.0]));
    }

    #[test]
    fn test_matrix_relu() {
        let a = Matrix {
            row: 2,
            col: 2,
            data: vec![-0.1, 2.0, 0.0, -1.0],
        };
        let act = a.relu();
        assert_eq!(act.data, vec![0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transpose_dot_vector() {
        let w = Matrix {
            row: 2,
            col: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let v = vec![1.0, 10.0];
        let result = w.transpose_dot_vector(&v).unwrap();
        // Expected:
        // result[0] = w[0][0]*v[0] + w[1][0]*v[1] = 1*1 + 4*10 = 41
        // result[1] = 2*1 + 5*10 = 52
        // result[2] = 3*1 + 6*10 = 63
        assert_eq!(result, vec![41.0, 52.0, 63.0]);
    }
}
