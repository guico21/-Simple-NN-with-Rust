/// Computes cross-entropy loss for a single sample with softmax.
///
/// # Arguments
/// * `logits` - raw scores from the network, length = number of classes (10 in this case for numbers ranging 0 -> 9
/// * `label`  - correct class index (0..9)
///
/// # Returns
/// * `(loss, gradient_wrt_logits)`
///   - `loss`: scalar (f32)
///   - `gradient_wrt_logits`: vector of same length as logits
/// ## Steps:
/// 1. Find the maximum logit – prevents overflow when exponentiating.
/// 2. Shift logits by subtracting the max, then compute `sum_exp = Σ exp(shifted)`.
/// 3. Compute `log_sum_exp = max_logit + ln(sum_exp)`.
///    This equals `ln(Σ exp(logits))` in a stable way.
/// 4. Calculate log‑probabilities → `log_probs[i] = logits[i] - log_sum_exp`.
///    These are exactly `ln(softmax(logits)[i])`.
/// 5. Loss = `-log_probs[label]` (negative log‑likelihood of the correct class).
/// 6. Probabilities = `exp(log_probs)`, i.e., the softmax output.
pub fn cross_entropy_loss(logits: &[f32], label: usize) -> (f32, Vec<f32>, Vec<f32>) {
    let mut max_logit = f32::NEG_INFINITY; // this is the smallest possible value in the IEEE 754 ordering
    for value in logits {
        max_logit = max_logit.max(*value);
    }
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let log_sum_exp = max_logit + sum_exp.ln();
    let loss = -logits[label] + log_sum_exp;
    let probs: Vec<f32> = logits
        .iter()
        .map(|&x| (x - max_logit).exp() / sum_exp)
        .collect();
    let mut grad = probs.clone(); // p vector
    grad[label] -= 1.0; // subtract one at the correct class
    (loss, probs, grad)
}

// test below written with AI
#[cfg(test)]
mod tests {
    use super::*;

    // Helper to compare floats with a tolerance
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // Verify that probabilities sum to 1
    fn check_probs_sum_to_one(probs: &[f32]) {
        let sum: f32 = probs.iter().sum();
        assert!(
            approx_eq(sum, 1.0, 1e-6),
            "probabilities sum to {}, not 1.0",
            sum
        );
    }

    #[test]
    fn test_uniform_logits() {
        // With all logits equal, each probability = 1/n, loss = ln(n)
        let logits = [2.0, 2.0, 2.0, 2.0];
        let n = logits.len() as f32;
        for label in 0..4 {
            let (loss, probs, _grad) = cross_entropy_loss(&logits, label); // <-- updated
            check_probs_sum_to_one(&probs);
            let expected_prob = 1.0 / n;
            for &p in &probs {
                assert!(approx_eq(p, expected_prob, 1e-6));
            }
            assert!(
                approx_eq(loss, n.ln(), 1e-6),
                "loss = {}, expected ln({}) = {}",
                loss,
                n,
                n.ln()
            );
        }
    }

    #[test]
    fn test_one_hot_like_logits() {
        // One logit dominates -> probability close to 1, loss close to 0 if correct
        let logits = [1.0, 10.0, -1.0];
        let (loss, probs, _grad) = cross_entropy_loss(&logits, 1); // <-- updated
        check_probs_sum_to_one(&probs);
        assert!(
            probs[1] > 0.999,
            "probability of dominant class should be near 1"
        );
        assert!(loss < 1e-3, "loss should be near 0, got {}", loss);

        // If label is wrong, loss should be high
        let (loss_wrong, _, _) = cross_entropy_loss(&logits, 0); // <-- updated
        assert!(
            loss_wrong > 5.0,
            "loss for wrong class should be large, got {}",
            loss_wrong
        );
    }

    #[test]
    fn test_large_logits_stability() {
        // Very large values should not cause overflow
        let logits = [1000.0, 1010.0, 990.0];
        let (loss, probs, _grad) = cross_entropy_loss(&logits, 1); // <-- updated
        check_probs_sum_to_one(&probs);
        assert!(probs[1] > 0.99);
        assert!(loss < 1e-3);
    }

    #[test]
    fn test_negative_logits() {
        let logits = [-5.0, -1.0, -3.0];
        let (loss, probs, _grad) = cross_entropy_loss(&logits, 1); // <-- updated
        check_probs_sum_to_one(&probs);
        // Softmax of [-5, -1, -3] after shifting: max -1 -> [-4,0,-2] -> exp: ~0.0183, 1, ~0.1353, sum=1.1536, probs: 0.0159, 0.867, 0.117
        assert!(approx_eq(probs[1], 0.867, 0.01));
        assert!(loss > 0.0);
    }

    #[test]
    fn test_loss_non_negative() {
        let logits = [0.1, 0.2, 0.3];
        for lbl in 0..3 {
            let (loss, _, _) = cross_entropy_loss(&logits, lbl); // <-- updated
            assert!(loss >= 0.0, "loss should be non-negative, got {}", loss);
        }
    }

    #[test]
    fn test_single_class() {
        // Degenerate case: only one class
        let logits = [42.0];
        let (loss, probs, _grad) = cross_entropy_loss(&logits, 0); // <-- updated
        assert_eq!(probs.len(), 1);
        assert!(approx_eq(probs[0], 1.0, 1e-6));
        assert!(approx_eq(loss, 0.0, 1e-6));
    }
}
