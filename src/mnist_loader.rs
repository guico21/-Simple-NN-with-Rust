pub struct MnistData {
    // also in this case, data are stored in a long vector. that is to ensure memory management is easier (for the laptop, not for me)
    pub images: Vec<f32>,
    pub labels: Vec<u8>,   // image label
    pub num_images: usize, // number of images. not sure if I will use it
}

/// Loads the data from the data files of the MNIST dataset, considering images a 28x28 (784) format over a vector.
pub fn load_mnist_data(dir: &str) -> Result<MnistData, String> {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for digit in 0..=9 {
        let path = format!("{}/data{}", dir, digit);
        // below we read the contents of a vile into a Vec<u8>
        // in case of failure we have Err(io_error)
        // .map_err( - ) if the error occurs, io::Error(e) is managed as a String
        let bytes = match std::fs::read(&path) {
            Ok(content) => content,
            Err(e) => return Err(format!("failed to read {}: {}", path, e)),
        };
        if bytes.len() != 1000 * 784 {
            // 784 = 28*28
            return Err(format!(
                "The file in {} has {} bytes. Expected {}.",
                path,
                bytes.len(),
                1000 * 784
            ));
        }
        // image conversion
        for chunk in bytes.chunks_exact(784) {
            images.extend(chunk.iter().map(|&b| b as f32 / 255.0));
            labels.push(digit);
        }
    }
    let num_images = labels.len();
    Ok(MnistData {
        images,
        labels,
        num_images,
    })
}
