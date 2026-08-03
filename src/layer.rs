use crate::{matrix::Matrix, utils::add_vector};
use rand::distr::{Distribution, Uniform};

pub enum Activation {
    ReLU,
    Linear, // before softmax
}

pub struct Layer {
    w: Matrix,
    b: Vec<f32>,
    activation: Activation,
    last_input: Option<Vec<f32>>,
    last_z: Option<Vec<f32>>,
    last_a: Option<Vec<f32>>,
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
                vec![0.0; input.len()]
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
}
