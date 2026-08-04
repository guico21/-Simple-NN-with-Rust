use std::fs::{self, File};
use std::io::Write;

use crate::network::Network;

pub fn save_model_weights(network: &Network, dir: &str) {
    fs::create_dir_all(dir).expect("Cannot create directory for storing model weights");
    for (i, layer) in network.layers.iter().enumerate() {
        let file_path = format!("{}/layer_{}.txt", dir, i);
        let mut f = File::create(&file_path).expect("Cannot create model file.");
        // write dimentions
        writeln!(f, "{} {}", layer.w.row, layer.w.col).unwrap();
        // Write weights (row‑major)
        for val in &layer.w.data {
            write!(f, "{} ", val).unwrap();
        }
        writeln!(f).unwrap();
        // Write biases
        for val in &layer.b {
            write!(f, "{} ", val).unwrap();
        }
        writeln!(f).unwrap();
    }
}
