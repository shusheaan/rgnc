use nalgebra::Vector3;

const G0: f64 = 9.80665;
const R_EARTH: f64 = 6_371_000.0;

/// Constant gravity model (surface-level approximation).
pub fn gravity_constant() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -G0)
}

/// Altitude-dependent gravity (inverse-square law).
pub fn gravity_altitude(altitude: f64) -> Vector3<f64> {
    let factor = (R_EARTH / (R_EARTH + altitude)).powi(2);
    Vector3::new(0.0, 0.0, -G0 * factor)
}
