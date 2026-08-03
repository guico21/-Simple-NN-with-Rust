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
    pub fn forward_pass(&mut self, input: &[f32]) {}
    pub fn cross_entropy_loss(logits: &[f32], label: usize) -> (f32, Vec<f32>) {
        (0.0, vec![1.0, 1.0])
    }
    pub fn backward_pass(&mut self, grad_output: &[f32]) {}
    pub fn update_weights(&mut self, learning_rate: f32) {}
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
}
