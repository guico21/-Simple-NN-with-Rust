use crate::{matrix::Matrix, utils::add_vector};
use rand::distr::{Distribution, Uniform};

pub enum Activation {
    ReLU,
    Linear, // before softmax
}

pub struct Layer {
    pub w: Matrix,
    pub b: Vec<f32>,
    pub activation: Activation,
    pub last_input: Option<Vec<f32>>,
    pub last_z: Option<Vec<f32>>,
    pub last_a: Option<Vec<f32>>,
    pub w_grad: Option<Matrix>,   // added for gradient
    pub b_grad: Option<Vec<f32>>, // added for gradient
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, activation: Activation) -> Layer {
        // He uniform initialization for ReLU
        let limit = (6.0_f32 / n_in as f32).sqrt();
        let uniform = Uniform::new(-limit, limit).expect("invalid range");
        let mut rng = rand::rng();
        let data = std::iter::repeat_with(|| uniform.sample(&mut rng))
            .take(n_out * n_in)
            .collect();
        let w = Matrix::new(n_out, n_in, Some(data));
        let b = vec![0.0; n_out];
        Layer {
            w: w,
            b: b,
            activation,
            last_input: None,
            last_z: None,
            last_a: None,
            w_grad: None,
            b_grad: None,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let z = match self.w.dot_product_vector(input) {
            Some(matrix) => matrix,
            None => {
                eprintln!("Warning: dot product between matrix and vector failed, returning zeros");
                vec![0.0; self.w.row]
            }
        };
        let z_bias = match add_vector(&z, &self.b) {
            Some(matrix) => matrix,
            None => {
                eprintln!("Warning: addign vector to a matrix failed, returning zeros");
                vec![0.0; self.w.row]
            }
        };
        self.last_input = Some(input.to_vec());
        self.last_z = Some(z_bias.clone());
        let a = match self.activation {
            Activation::ReLU => z_bias.iter().map(|&x| x.max(0.0)).collect(),
            Activation::Linear => z_bias.clone(),
        };
        self.last_a = Some(a.clone());
        a
    }

    pub fn backward(&mut self, d_out: &[f32]) -> Vec<f32> {
        // get caches. there is a panic if forward was not called before
        // thus caches will be None.
        // WARN must improve the erro handling here
        let z = match &self.last_z {
            Some(z) => z,
            None => panic!("Forward not called due to last_z."),
        };
        let input = match &self.last_input {
            Some(z) => z,
            None => panic!("Forward not called due to last_input"),
        };
        // activation backward
        let d_z = match self.activation {
            Activation::ReLU => d_out
                .iter()
                .zip(z.iter())
                .map(|(&do_val, &z_val)| if z_val > 0.0 { do_val } else { 0.0 })
                .collect(),
            Activation::Linear => d_out.to_vec(),
        };
        // weight gradient
        let n_in = self.w.col;
        let n_out = self.w.row;
        let mut w_grad_data = vec![0.0; n_out * n_in];
        for r in 0..n_out {
            for c in 0..n_in {
                w_grad_data[r * n_in + c] = d_z[r] * input[c];
            }
        }
        self.w_grad = Some(Matrix::new(n_out, n_in, Some(w_grad_data)));
        // bias gradient
        self.b_grad = Some(d_z.clone());
        // gradient with respect the input (for previous layer)
        let d_prev = self
            .w
            .transpose_dot_vector(&d_z)
            .expect("Dimension mismatch in backward.");
        d_prev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_layer() {
        let layer = Layer::new(4, 2, Activation::ReLU);
        // Matrix should have 2 rows (n_out) and 4 cols (n_in)
        assert_eq!(layer.w.row, 2);
        assert_eq!(layer.w.col, 4);
        assert_eq!(layer.w.data.len(), 8);
        // Bias should have length 2 (n_out)
        assert_eq!(layer.b.len(), 2);
        // Last values should be None initially
        assert!(layer.last_input.is_none());
        assert!(layer.last_z.is_none());
        assert!(layer.last_a.is_none());
        assert!(layer.w_grad.is_none());
        assert!(layer.b_grad.is_none());
    }

    #[test]
    fn test_forward_relu() {
        // Manually construct a layer with 3 inputs and 2 outputs
        // Weights matrix (2x3):
        // [1.0, 2.0,  3.0]
        // [-1.0, -2.0, -3.0]
        let mut layer = Layer {
            w: Matrix::new(2, 3, Some(vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])),
            b: vec![0.5, -0.5],
            activation: Activation::ReLU,
            last_input: None,
            last_z: None,
            last_a: None,
            w_grad: None,
            b_grad: None,
        };
        let input = vec![1.0, 1.0, 1.0];
        let output = layer.forward(&input);
        // z = W * x + b
        // z[0] = (1*1 + 2*1 + 3*1) + 0.5 = 6.5
        // z[1] = (-1*1 - 2*1 - 3*1) - 0.5 = -6.5
        // ReLU(z) -> [6.5, 0.0]
        assert_eq!(output, vec![6.5, 0.0]);
        // Check if caches were properly stored
        assert_eq!(layer.last_input, Some(vec![1.0, 1.0, 1.0]));
        assert_eq!(layer.last_z, Some(vec![6.5, -6.5]));
        assert_eq!(layer.last_a, Some(vec![6.5, 0.0]));
    }

    #[test]
    fn test_forward_linear() {
        let mut layer = Layer {
            w: Matrix::new(2, 3, Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
            b: vec![1.0, 2.0],
            activation: Activation::Linear,
            last_input: None,
            last_z: None,
            last_a: None,
            w_grad: None,
            b_grad: None,
        };
        let input = vec![4.0, 5.0, 6.0];
        let output = layer.forward(&input);
        // z = W * x + b
        // z[0] = (1*4 + 0*5 + 0*6) + 1.0 = 5.0
        // z[1] = (0*4 + 1*5 + 0*6) + 2.0 = 7.0
        // Linear -> [5.0, 7.0]
        assert_eq!(output, vec![5.0, 7.0]);
        assert_eq!(layer.last_z, Some(vec![5.0, 7.0]));
    }

    #[test]
    fn test_backward_gradients() {
        // Manually construct a layer (2 outputs, 3 inputs) with ReLU activation.
        // Weights (2x3):
        // [ 1.0,  2.0,  3.0]
        // [-1.0, -2.0, -3.0]
        let mut layer = Layer {
            w: Matrix::new(2, 3, Some(vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0])),
            b: vec![0.5, -0.5],
            activation: Activation::ReLU,
            last_input: None,
            last_z: None,
            last_a: None,
            w_grad: None,
            b_grad: None,
        };
        let input = vec![1.0, 2.0, -1.0];
        let d_out = vec![1.0, -1.0];

        // Forward then backward to populate caches and analytic gradients.
        let _ = layer.forward(&input);
        let _ = layer.backward(&d_out);

        // Snapshot the analytic gradients (clone so we can mutate `layer`
        // during the finite-difference loop below without borrow conflicts).
        let analytic_w = layer.w_grad.as_ref().expect("w_grad not set").data.clone();
        let analytic_b = layer.b_grad.as_ref().expect("b_grad not set").clone();

        // Treat the layer output `a` as feeding into a scalar loss
        //   L(a) = sum_i d_out[i] * a[i]
        // so that dL/da = d_out. The numerical gradient of L with respect to
        // each weight/bias should match the stored analytic gradients.
        let eps = 1e-3_f32; // oringal was 1e-4, but with rounding on f32, we kept receiving 1.001358 making the error
        let n_out = layer.w.row;
        let n_in = layer.w.col;
        let loss =
            |a: &[f32], d: &[f32]| -> f32 { a.iter().zip(d.iter()).map(|(ai, di)| ai * di).sum() };

        // Numerical w_grad via central difference.
        let mut num_w = vec![0.0; n_out * n_in];
        for r in 0..n_out {
            for c in 0..n_in {
                let idx = r * n_in + c;
                let orig = layer.w.data[idx];

                layer.w.data[idx] = orig + eps;
                let l_plus = loss(&layer.forward(&input), &d_out);

                layer.w.data[idx] = orig - eps;
                let l_minus = loss(&layer.forward(&input), &d_out);

                layer.w.data[idx] = orig; // restore
                num_w[idx] = (l_plus - l_minus) / (2.0 * eps);
            }
        }

        // Numerical b_grad via central difference.
        let mut num_b = vec![0.0; n_out];
        for i in 0..n_out {
            let orig = layer.b[i];

            layer.b[i] = orig + eps;
            let l_plus = loss(&layer.forward(&input), &d_out);

            layer.b[i] = orig - eps;
            let l_minus = loss(&layer.forward(&input), &d_out);

            layer.b[i] = orig; // restore
            num_b[i] = (l_plus - l_minus) / (2.0 * eps);
        }

        // Compare analytic vs numerical within tolerance.
        // Note: with input=[1,2,-1], b=[0.5,-0.5], z=[2.5,-2.5] — safely away
        // from the ReLU kink at 0, so finite differences should be accurate.
        for (i, (a, n)) in analytic_w.iter().zip(num_w.iter()).enumerate() {
            assert!(
                (a - n).abs() < 1e-3,
                "w_grad mismatch at [{}]: analytic={}, numerical={}",
                i,
                a,
                n
            );
        }
        for (i, (a, n)) in analytic_b.iter().zip(num_b.iter()).enumerate() {
            assert!(
                (a - n).abs() < 1e-3,
                "b_grad mismatch at [{}]: analytic={}, numerical={}",
                i,
                a,
                n
            );
        }
    }
}
