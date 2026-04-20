/// Aerodynamic coefficient lookup table with bilinear interpolation.

/// Lookup table for Cd, Cl, and Cy as functions of Mach and angle.
#[derive(Debug, Clone)]
pub struct AeroTable {
    pub mach_breaks: Vec<f64>,
    pub alpha_breaks: Vec<f64>,
    pub cd: Vec<Vec<f64>>,
    pub cl: Vec<Vec<f64>>,
    /// Side-force coefficient Cy(Mach, beta).
    pub cy: Vec<Vec<f64>>,
}

impl AeroTable {
    /// Create a constant-Cd table (no Mach/alpha dependence, zero Cl and Cy).
    pub fn constant(cd: f64) -> Self {
        Self {
            mach_breaks: vec![0.0, 10.0],
            alpha_breaks: vec![0.0, 1.0],
            cd: vec![vec![cd, cd], vec![cd, cd]],
            cl: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            cy: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        }
    }

    /// Lookup Cd and Cl with bilinear interpolation.
    pub fn lookup(&self, mach: f64, alpha: f64) -> (f64, f64) {
        let cd = interp2d(&self.mach_breaks, &self.alpha_breaks, &self.cd, mach, alpha);
        let cl = interp2d(&self.mach_breaks, &self.alpha_breaks, &self.cl, mach, alpha);
        (cd, cl)
    }

    /// Lookup side-force coefficient Cy(Mach, beta).
    pub fn lookup_cy(&self, mach: f64, beta: f64) -> f64 {
        interp2d(&self.mach_breaks, &self.alpha_breaks, &self.cy, mach, beta)
    }
}

fn interp2d(xs: &[f64], ys: &[f64], zs: &[Vec<f64>], x: f64, y: f64) -> f64 {
    if xs.len() < 2 || ys.len() < 2 {
        return zs[0][0];
    }

    let ix = find_interval(xs, x);
    let iy = find_interval(ys, y);

    let x0 = xs[ix];
    let x1 = xs[ix + 1];
    let y0 = ys[iy];
    let y1 = ys[iy + 1];

    let tx = if (x1 - x0).abs() > 1e-15 { (x - x0) / (x1 - x0) } else { 0.0 };
    let ty = if (y1 - y0).abs() > 1e-15 { (y - y0) / (y1 - y0) } else { 0.0 };

    let tx = tx.clamp(0.0, 1.0);
    let ty = ty.clamp(0.0, 1.0);

    let z00 = zs[ix][iy];
    let z10 = zs[ix + 1][iy];
    let z01 = zs[ix][iy + 1];
    let z11 = zs[ix + 1][iy + 1];

    (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
}

fn find_interval(breaks: &[f64], val: f64) -> usize {
    if val <= breaks[0] {
        return 0;
    }
    for i in 0..breaks.len() - 1 {
        if val < breaks[i + 1] {
            return i;
        }
    }
    breaks.len() - 2
}
