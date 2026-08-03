/// Adds two vectors of same size together. The vectors are N x 1.
pub fn add_vector(vector_1: &[f32], vector_2: &[f32]) -> Option<Vec<f32>> {
    if vector_1.len() != vector_2.len() {
        return None;
    }
    let result = vector_1
        .iter()
        .zip(vector_2.iter())
        .map(|(&a, &b)| a + b) // tuple cause of zip method.
        .collect();
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vector() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0];
        let result = add_vector(&a, &b);
        assert_eq!(
            result,
            Some(vec![11.0, 13.0, 15.0, 17.0]),
            "The sum of two Nx1 vectors is NOT successfull."
        );
    }
}
