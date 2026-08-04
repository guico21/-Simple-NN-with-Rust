use crate::layer::{Activation, Layer};

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: &[usize]) -> Network {
        if layer_sizes.len() < 2 {
            assert!(
                layer_sizes.len() >= 2,
                "Minimun two layers are requried. Provided less than 2."
            );
        }
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..(layer_sizes.len() - 1) {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];
            // Everything except the last layer gets ReLU;
            // the final layer is linear (raw logits).
            let activation = if i == (layer_sizes.len() - 2) {
                Activation::Linear
            } else {
                Activation::ReLU
            };
            layers.push(Layer::new(n_in, n_out, activation));
        }
        Network { layers }
    }

    pub fn forward_pass(&mut self, input: &[f32]) -> Vec<f32> {
        let mut activation = input.to_vec(); // <- #WARN I am copying memory here
        for layer in self.layers.iter_mut() {
            activation = layer.forward(&activation);
        }
        activation
    }

    pub fn backward_pass(&mut self, grad_output: &[f32]) {
        let mut grad = grad_output.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    pub fn update_weights(&mut self, learning_rate: f32) {
        for layer in self.layers.iter_mut() {
            // weight update
            if let Some(w_grad) = &layer.w_grad {
                for (w, &gw) in layer.w.data.iter_mut().zip(w_grad.data.iter()) {
                    *w = *w - (learning_rate * gw);
                }
            }
            // bias update
            if let Some(b_grad) = &layer.b_grad {
                for (b, &gb) in layer.b.iter_mut().zip(b_grad.iter()) {
                    *b = *b - (learning_rate * gb);
                }
            }
            // clearing gradients. they will be recalculated the next backward pass
            layer.w_grad = None;
            layer.b_grad = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_construction() {
        let net = Network::new(&[784, 128, 64, 10]);
        assert_eq!(net.layers.len(), 3);

        // Layer 0: 784 -> 128, ReLU
        assert_eq!(net.layers[0].w.row, 128);
        assert_eq!(net.layers[0].w.col, 784);
        assert!(matches!(net.layers[0].activation, Activation::ReLU));

        // Layer 1: 128 -> 64, ReLU
        assert_eq!(net.layers[1].w.row, 64);
        assert_eq!(net.layers[1].w.col, 128);
        assert!(matches!(net.layers[1].activation, Activation::ReLU));

        // Layer 2: 64 -> 10, Linear
        assert_eq!(net.layers[2].w.row, 10);
        assert_eq!(net.layers[2].w.col, 64);
        assert!(matches!(net.layers[2].activation, Activation::Linear));
    }

    #[test]
    fn test_forward_pass() {
        // 2 inputs -> 3 hidden (ReLU) -> 1 output (Linear)
        let mut net = Network::new(&[2, 3, 1]);
        // Overwrite weights with deterministic values (all 1.0) and biases 0.0
        net.layers[0].w.data = vec![1.0; 3 * 2];
        net.layers[0].b = vec![0.0; 3];
        // Layer 1: 1x3 matrix of all 1.0
        net.layers[1].w.data = vec![1.0; 1 * 3];
        net.layers[1].b = vec![0.0; 1];
        let input = vec![1.0, 2.0];
        let output = net.forward_pass(&input);
        // Expected:
        // Hidden layer z = [1*1+1*2=3.0, 3.0, 3.0] + 0.0 = [3,3,3] -> ReLU -> [3,3,3]
        // Output layer z = [1*3+1*3+1*3=9.0] + 0.0 = [9.0] -> Linear -> [9.0]
        assert_eq!(output, vec![9.0]);
        // Verify caches on the hidden layer (index 0)
        let hidden = &net.layers[0];
        assert_eq!(hidden.last_input.as_ref().unwrap(), &vec![1.0, 2.0]);
        assert_eq!(hidden.last_z.as_ref().unwrap(), &vec![3.0, 3.0, 3.0]);
        assert_eq!(hidden.last_a.as_ref().unwrap(), &vec![3.0, 3.0, 3.0]);
        // Verify caches on the output layer (index 1)
        let out_layer = &net.layers[1];
        assert_eq!(out_layer.last_input.as_ref().unwrap(), &vec![3.0, 3.0, 3.0]);
        assert_eq!(out_layer.last_z.as_ref().unwrap(), &vec![9.0]);
        assert_eq!(out_layer.last_a.as_ref().unwrap(), &vec![9.0]);
    }
}
