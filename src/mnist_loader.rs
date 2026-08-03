// TODO: Make the values of images and size parametric. right now is all hard coded.

pub struct MnistData {
    // also in this case, data are stored in a long vector. that is to ensure memory management is easier (for the laptop, not for me)
    pub images: Vec<f32>, // length = 10_000 * 784 = 7_840_000 for the MNIST dataset
    pub labels: Vec<u8>,  // image label
}

impl MnistData {
    /// Returns a 28x28 image given the index.
    pub fn get_image(&self, i: usize) -> &[f32] {
        let start = i * 784;
        &self.images[start..start + 784]
    }
}

/// Loads the data from the data files of the MNIST dataset, considering images a 28x28 (784) format over a vector.
/// The info from the website mentions that images are stored as un
/// ## File Format
/// Each file has 1000 training examples. Each training example is of size 28x28 pixels.
/// The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255.
/// The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.
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
    Ok(MnistData { images, labels })
}
