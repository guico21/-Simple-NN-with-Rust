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
pub fn cross_entropy_loss(logits: &[f32], label: usize) -> (f32, Vec<f32>) {
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
    (loss, probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // test written by AI
    //
}
