/// Applies an IIR filter in-place, writing results to `output`.
///
/// `numerator_coeffs` and `denom_coeffs` are length-3 coefficient arrays.
/// `init_conditions` is a 2-element state that is updated across calls.
/// `output` must be at least as long as `signal`.
#[inline(always)]
pub fn filter_signal_into(
    numerator_coeffs: &[f64; 3],
    denom_coeffs: &[f64; 3],
    signal: &[f64],
    output: &mut [f64],
    init_conditions: &mut [f64; 2],
) {
    let n0 = numerator_coeffs[0];
    let n1 = numerator_coeffs[1];
    let n2 = numerator_coeffs[2];
    let d1 = denom_coeffs[1];
    let d2 = denom_coeffs[2];
    let mut c0 = init_conditions[0];
    let mut c1 = init_conditions[1];

    for (out, &s) in output.iter_mut().zip(signal.iter()) {
        let filtered = n0 * s + c0;
        c0 = n1 * s + c1 - d1 * filtered;
        c1 = n2 * s - d2 * filtered;
        *out = filtered;
    }

    init_conditions[0] = c0;
    init_conditions[1] = c1;
}

