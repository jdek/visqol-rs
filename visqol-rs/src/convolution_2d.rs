use ndarray::{Array2, ShapeBuilder};
use std::simd::f64x4;

/// Computes the 2D convolution of `input_matrix` with `fir_filter`, using
/// boundary reflection (replicate-padding). The output has the same dimensions
/// as `input_matrix`.
///
/// This is a general implementation that falls through to a specialized 3×3
/// fast-path when the filter is 3×3 (the only size used in ViSQOL).
pub fn perform_valid_2d_conv_with_boundary(
    fir_filter: &Array2<f64>,
    input_matrix: &Array2<f64>,
) -> Array2<f64> {
    let f_r = fir_filter.nrows();
    let f_c = fir_filter.ncols();

    if f_r == 3 && f_c == 3 {
        return conv2d_3x3_reflected(fir_filter, input_matrix);
    }

    // Fallback: general case (kept for correctness, not on hot path)
    let mut input_copy = input_matrix.clone();
    let padded_matrix = add_matrix_boundary(&mut input_copy);
    let i_r_c = padded_matrix.nrows();
    let i_c_c = padded_matrix.ncols();
    let o_r_c = i_r_c - f_r + 1;
    let o_c_c = i_c_c - f_c + 1;

    let mut out_matrix = Array2::<f64>::zeros((o_r_c, o_c_c).f());

    for o_row in 0..o_r_c {
        for o_col in 0..o_c_c {
            let mut sum = 0.0f64;
            for f_row in 0..f_r {
                for f_col in 0..f_c {
                    let ir = o_row + f_row;
                    let ic = o_col + f_col;
                    // Filter is applied in reverse order (correlation-style)
                    let fr = f_r - 1 - f_row;
                    let fc = f_c - 1 - f_col;
                    sum += padded_matrix[(ir, ic)] * fir_filter[(fr, fc)];
                }
            }
            out_matrix[(o_row, o_col)] = sum;
        }
    }
    out_matrix
}

/// Specialized 3×3 convolution with replicate-boundary padding.
/// Computes boundary reflections on-the-fly, avoiding all intermediate
/// matrix allocations. Uses raw slice access and SIMD for the interior.
#[inline(always)]
fn conv2d_3x3_reflected(filter: &Array2<f64>, input: &Array2<f64>) -> Array2<f64> {
    let nrows = input.nrows();
    let ncols = input.ncols();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    conv2d_3x3_reflected_core(filter, input, &mut out);
    out
}

/// Same as `conv2d_3x3_reflected` but writes into a pre-allocated output.
#[inline(always)]
pub fn conv2d_3x3_reflected_into(filter: &Array2<f64>, input: &Array2<f64>, out: &mut Array2<f64>) {
    debug_assert_eq!(input.nrows(), out.nrows());
    debug_assert_eq!(input.ncols(), out.ncols());
    conv2d_3x3_reflected_core(filter, input, out);
}

#[inline(always)]
fn conv2d_3x3_reflected_core(filter: &Array2<f64>, input: &Array2<f64>, out: &mut Array2<f64>) {
    let nrows = input.nrows();
    let ncols = input.ncols();

    // Reversed filter weights (convolution = cross-correlation with flipped kernel)
    let w00 = filter[(2, 2)];
    let w01 = filter[(2, 1)];
    let w02 = filter[(2, 0)];
    let w10 = filter[(1, 2)];
    let w11 = filter[(1, 1)];
    let w12 = filter[(1, 0)];
    let w20 = filter[(0, 2)];
    let w21 = filter[(0, 1)];
    let w22 = filter[(0, 0)];

    // Ensure input is contiguous so we can use raw slice access.
    let input_c = if input.is_standard_layout() {
        None
    } else {
        Some(input.as_standard_layout().into_owned())
    };
    let inp = input_c.as_ref().unwrap_or(input);
    let inp_slice = inp.as_slice().expect("input not contiguous after layout fix");

    let out_slice = out.as_slice_mut().expect("output not contiguous");

    // SIMD kernel weights
    let sw00 = f64x4::splat(w00);
    let sw01 = f64x4::splat(w01);
    let sw02 = f64x4::splat(w02);
    let sw10 = f64x4::splat(w10);
    let sw11 = f64x4::splat(w11);
    let sw12 = f64x4::splat(w12);
    let sw20 = f64x4::splat(w20);
    let sw21 = f64x4::splat(w21);
    let sw22 = f64x4::splat(w22);

    for r in 0..nrows {
        let rm1 = if r == 0 { 0 } else { r - 1 };
        let rp1 = if r + 1 >= nrows { nrows - 1 } else { r + 1 };

        let row_m1 = rm1 * ncols;
        let row_0 = r * ncols;
        let row_p1 = rp1 * ncols;

        // First column (c=0): boundary
        {
            let cm1 = 0;
            let cp1 = if ncols > 1 { 1 } else { 0 };

            out_slice[row_0] = w00 * inp_slice[row_m1 + cm1]
                + w01 * inp_slice[row_m1]
                + w02 * inp_slice[row_m1 + cp1]
                + w10 * inp_slice[row_0 + cm1]
                + w11 * inp_slice[row_0]
                + w12 * inp_slice[row_0 + cp1]
                + w20 * inp_slice[row_p1 + cm1]
                + w21 * inp_slice[row_p1]
                + w22 * inp_slice[row_p1 + cp1];
        }

        // Interior columns with SIMD: process 4 at a time
        let interior_end = ncols.saturating_sub(1);
        let mut c = 1usize;
        while c + 4 <= interior_end {
            // Load 6 values per row (c-1..c+4) to cover 4 outputs
            let load_m1 = |off: usize| f64x4::from_slice(&inp_slice[row_m1 + off..]);
            let load_0 = |off: usize| f64x4::from_slice(&inp_slice[row_0 + off..]);
            let load_p1 = |off: usize| f64x4::from_slice(&inp_slice[row_p1 + off..]);

            let cm1 = c - 1;

            let result = sw00 * load_m1(cm1)
                + sw01 * load_m1(c)
                + sw02 * load_m1(c + 1)
                + sw10 * load_0(cm1)
                + sw11 * load_0(c)
                + sw12 * load_0(c + 1)
                + sw20 * load_p1(cm1)
                + sw21 * load_p1(c)
                + sw22 * load_p1(c + 1);

            let arr = result.to_array();
            out_slice[row_0 + c] = arr[0];
            out_slice[row_0 + c + 1] = arr[1];
            out_slice[row_0 + c + 2] = arr[2];
            out_slice[row_0 + c + 3] = arr[3];
            c += 4;
        }

        // Scalar tail for remaining interior columns
        while c < interior_end {
            let cm1 = c - 1;
            let cp1 = c + 1;

            out_slice[row_0 + c] = w00 * inp_slice[row_m1 + cm1]
                + w01 * inp_slice[row_m1 + c]
                + w02 * inp_slice[row_m1 + cp1]
                + w10 * inp_slice[row_0 + cm1]
                + w11 * inp_slice[row_0 + c]
                + w12 * inp_slice[row_0 + cp1]
                + w20 * inp_slice[row_p1 + cm1]
                + w21 * inp_slice[row_p1 + c]
                + w22 * inp_slice[row_p1 + cp1];
            c += 1;
        }

        // Last column (if ncols > 1)
        if ncols > 1 {
            let c = ncols - 1;
            let cm1 = c - 1;
            let cp1 = c; // clamped

            out_slice[row_0 + c] = w00 * inp_slice[row_m1 + cm1]
                + w01 * inp_slice[row_m1 + c]
                + w02 * inp_slice[row_m1 + cp1]
                + w10 * inp_slice[row_0 + cm1]
                + w11 * inp_slice[row_0 + c]
                + w12 * inp_slice[row_0 + cp1]
                + w20 * inp_slice[row_p1 + cm1]
                + w21 * inp_slice[row_p1 + c]
                + w22 * inp_slice[row_p1 + cp1];
        }
    }
}

/// Compute zero-padded matrix and fill zero-padded boundaries with the adjacent non-zero rows and columns
pub fn add_matrix_boundary(input_matrix: &mut Array2<f64>) -> Array2<f64> {
    let mut output_matrix = copy_matrix_within_padding(input_matrix, 1, 1, 1, 1);

    for i in 0..output_matrix.ncols() {
        output_matrix.row_mut(0)[i] = output_matrix.row(1)[i];
        output_matrix.row_mut(output_matrix.nrows() - 1)[i] =
            output_matrix.row(output_matrix.nrows() - 2)[i];
    }

    for i in 0..output_matrix.nrows() {
        output_matrix.column_mut(0)[i] = output_matrix.column_mut(1)[i];
        output_matrix.column_mut(output_matrix.ncols() - 1)[i] =
            output_matrix.column(output_matrix.ncols() - 2)[i];
    }
    output_matrix
}

/// Returns a copy of `input matrix` which is zero-padded by the specified amounts.
pub fn copy_matrix_within_padding(
    input_matrix: &Array2<f64>,
    row_prepad_amount: usize,
    row_postpad_amount: usize,
    col_prepad_amount: usize,
    col_postpad_amount: usize,
) -> Array2<f64> {
    let mut output_matrix = Array2::<f64>::zeros((
        row_prepad_amount + input_matrix.nrows() + row_postpad_amount,
        col_prepad_amount + input_matrix.ncols() + col_postpad_amount,
    ));

    for row_i in 0..input_matrix.nrows() {
        for col_i in 0..input_matrix.ncols() {
            output_matrix[(row_i + row_prepad_amount, col_i + col_prepad_amount)] =
                input_matrix[(row_i, col_i)];
        }
    }
    output_matrix
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, Array2, ShapeBuilder};

    use super::*;

    #[test]
    fn convolve_with_window() {
        let w = vec![
            0.0113033910173052,
            0.0838251475442633,
            0.0113033910173052,
            0.0838251475442633,
            0.619485845753726,
            0.0838251475442633,
            0.0113033910173052,
            0.0838251475442633,
            0.0113033910173052,
        ];
        let window = Array::from_shape_vec((3, 3).f(), w).unwrap();

        let m = vec![
            40.0392, 43.3409, 39.5270, 41.1731, 41.3591, 42.6852, 45.2083, 45.7769, 39.9689,
            43.6190, 41.0119, 40.4244, 41.5932, 43.6027, 42.6204, 43.0624, 42.2610, 42.4725,
            43.4258, 42.9079,
        ];
        let matrix = Array::from_shape_vec((5, 4).f(), m).unwrap();

        let result = perform_valid_2d_conv_with_boundary(&window, &matrix);

        let r = vec![
            40.6634, 42.8407, 40.6395, 41.0129, 41.5407, 42.4677, 44.2760, 44.2031, 41.2263,
            42.9752, 41.3784, 41.2656, 42.1388, 43.0366, 42.8042, 42.7613, 42.1817, 42.4590,
            43.2709, 42.9377,
        ];
        let expected_result = Array2::<f64>::from_shape_vec((5, 4).f(), r).unwrap();

        use approx::assert_abs_diff_eq;
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                assert_abs_diff_eq!(result[(i, j)], expected_result[(i, j)], epsilon = 0.001);
            }
        }
    }

    #[test]
    fn perform_padding() {
        let m = vec![
            40.0392, 43.3409, 39.5270, 41.1731, 41.3591, 42.6852, 45.2083, 45.7769, 39.9689,
            43.6190, 41.0119, 40.4244, 41.5932, 43.6027, 42.6204, 43.0624, 42.2610, 42.4725,
            43.4258, 42.9079,
        ];
        let mut matrix = Array::from_shape_vec((5, 4).f(), m).unwrap();
        let result = add_matrix_boundary(&mut matrix);

        let mut r = Vec::new();
        for i in 0..result.dim().0 {
            for j in 0..result.dim().1 {
                r.push(result[(i, j)]);
            }
        }

        let expected_result = vec![
            40.0392, 40.0392, 42.6852, 41.0119, 43.0624, 43.0624, 40.0392, 40.0392, 42.6852,
            41.0119, 43.0624, 43.0624, 43.3409, 43.3409, 45.2083, 40.4244, 42.261, 42.261, 39.527,
            39.527, 45.7769, 41.5932, 42.4725, 42.4725, 41.1731, 41.1731, 39.9689, 43.6027,
            43.4258, 43.4258, 41.3591, 41.3591, 43.619, 42.6204, 42.9079, 42.9079, 41.3591,
            41.3591, 43.619, 42.6204, 42.9079, 42.9079,
        ];

        assert_eq!(r, expected_result);
    }

    #[test]
    fn copy_with_zeros() {
        let m = vec![
            40.0392, 43.3409, 39.5270, 41.1731, 41.3591, 42.6852, 45.2083, 45.7769, 39.9689,
            43.6190, 41.0119, 40.4244, 41.5932, 43.6027, 42.6204, 43.0624, 42.2610, 42.4725,
            43.4258, 42.9079,
        ];
        let matrix = Array::from_shape_vec((5, 4).f(), m).unwrap();
        let result = copy_matrix_within_padding(&matrix, 1, 1, 1, 1);

        let er = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0392, 42.6852, 41.0119, 43.0624, 0.0, 0.0,
            43.3409, 45.2083, 40.4244, 42.261, 0.0, 0.0, 39.527, 45.7769, 41.5932, 42.4725, 0.0,
            0.0, 41.1731, 39.9689, 43.6027, 43.4258, 0.0, 0.0, 41.3591, 43.619, 42.6204, 42.9079,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // Extracted from cpp, with armadillo using column memory layout.
        let erm = Array::from_shape_vec((7, 6), er).unwrap();

        for (r_elem, erm_elem) in result.iter().zip(&erm) {
            assert_eq!(r_elem, erm_elem);
        }
    }
}
