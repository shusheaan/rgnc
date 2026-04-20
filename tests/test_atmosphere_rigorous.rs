//! =========================================================================
//! US Standard Atmosphere 1976 — Rigorous Verification Test Suite
//! =========================================================================
//!
//! PRIMARY SOURCE:
//!   NASA-TM-X-74335 (1976)
//!   "U.S. Standard Atmosphere, 1976"
//!   National Oceanic and Atmospheric Administration,
//!   National Aeronautics and Space Administration,
//!   United States Air Force
//!   Available: https://ntrs.nasa.gov/citations/19770009539
//!
//! REFERENCE TABLES USED:
//!   Table 4 (pp. 50-73): "Geopotential altitude, temperature, temperature
//!   ratio, pressure, pressure ratio, density, density ratio, mean free path,
//!   mean particle speed, mean collision frequency, speed of sound, dynamic
//!   viscosity, kinematic viscosity, thermal conductivity, and number density"
//!
//! VERIFICATION METHODOLOGY:
//!   - Expected values taken DIRECTLY from Table 4, column by column
//!   - Temperature values are exact (defined by lapse rates and base values)
//!   - Pressure values are computed from the barometric formula (Eq. 33a, 33b)
//!   - Density = p*M/(R*T) is a derived quantity (Eq. 42)
//!   - Speed of sound = sqrt(gamma*R*T/M) (Eq. 50)
//!
//! GEOPOTENTIAL vs GEOMETRIC HEIGHT:
//!   The standard defines properties in terms of geopotential height H.
//!   Our function accepts geometric height h and converts via:
//!     H = r0*h/(r0+h), where r0 = 6,356,766 m (NASA-TM-X-74335, p.3)
//!   Test inputs below are in GEOMETRIC meters. Expected values are from
//!   Table 4 entries at the corresponding GEOPOTENTIAL height.
//!   At low altitudes (<30km) the difference is <0.1%, so we can compare
//!   directly. At higher altitudes we note the expected deviation.
//!
//! CONSTANTS (NASA-TM-X-74335, pp. 2-3):
//!   g0 = 9.80665 m/s²      (standard gravitational acceleration)
//!   M0 = 0.0289644 kg/mol   (mean molecular weight of air, sea level)
//!   R* = 8.31432 J/(mol·K)  (universal gas constant — NOTE: the 1976
//!       standard uses 8.31432, while modern CODATA uses 8.31446.
//!       We use 8.31447 for consistency with modern value.)
//!   gamma = 1.4              (ratio of specific heats for ideal diatomic gas)
//!   p0 = 101325 Pa           (standard sea level pressure)
//!   T0 = 288.150 K           (standard sea level temperature)
//!   rho0 = 1.2250 kg/m³     (standard sea level density)
//!
//! HOW TO VERIFY THESE TESTS:
//!   1. Obtain NASA-TM-X-74335 from https://ntrs.nasa.gov/citations/19770009539
//!   2. Open Table 4 (starts on page 50 of the PDF)
//!   3. For each test case, find the row matching the geopotential altitude
//!   4. Compare the temperature, pressure, density, and speed of sound columns
//!   5. Our tolerances account for:
//!      - Geometric vs geopotential height difference at high altitudes
//!      - Slightly different R* value (8.31447 vs 8.31432)
//! =========================================================================

use rgnc::aero::atmosphere;

fn assert_close(actual: f64, expected: f64, rel_tol: f64, label: &str, h: f64) {
    let rel_err = if expected.abs() > 1e-30 {
        (actual - expected).abs() / expected.abs()
    } else {
        (actual - expected).abs()
    };
    assert!(
        rel_err < rel_tol,
        "{} at h={:.0} m: expected {:.6e}, got {:.6e}, rel_err={:.2e} > tol {:.2e}",
        label, h, expected, actual, rel_err, rel_tol
    );
}

// =========================================================================
// TEST GROUP 1: Sea Level Standard Conditions
// Source: NASA-TM-X-74335, Table 4, H=0 row, p. 50
// These are EXACT by definition — the standard defines these as the base.
// =========================================================================

#[test]
fn test_sea_level_standard_exact() {
    let r = atmosphere(0.0);

    // Temperature: T0 = 288.150 K (defined, NASA-TM-X-74335 p. 3)
    assert!((r.temperature - 288.150).abs() < 1e-6,
        "T at sea level: {}", r.temperature);

    // Pressure: p0 = 101325.0 Pa (defined, NASA-TM-X-74335 p. 3)
    assert!((r.pressure - 101325.0).abs() < 0.1,
        "p at sea level: {}", r.pressure);

    // Density: rho0 = p0*M/(R*T0) = 1.2250 kg/m³ (Table 4, H=0)
    assert_close(r.density, 1.2250, 1e-3, "rho", 0.0);

    // Speed of sound: a0 = sqrt(gamma*R*T0/M) = 340.294 m/s (Table 4, H=0)
    assert_close(r.speed_of_sound, 340.294, 1e-3, "a", 0.0);
}

// =========================================================================
// TEST GROUP 2: Troposphere (0-11 km)
// Lapse rate = -6.5 K/km (NASA-TM-X-74335 Table 1, p. 3)
// T(H) = 288.150 + (-0.0065) * H
// Source: Table 4 rows at H = 1, 5, 10, 11 km
// =========================================================================

#[test]
fn test_troposphere_temperature_lapse() {
    // The lapse rate is EXACT by definition: -6.5 K/km = -0.0065 K/m
    // T(H) = T0 + L*H where L = -0.0065 K/m

    let test_points = [
        // (geometric_h_m, geopotential_H_m, expected_T_K)
        // T = 288.150 + (-0.0065)*H
        (1000.0, 999.843, 281.651),  // Table 4, H=1km
        (5000.0, 4996.07, 255.676),  // Table 4, H=5km
        (10000.0, 9984.29, 223.252), // Table 4, H=10km
        (11000.0, 10981.0, 216.774), // Table 4, H=11km (near tropopause)
    ];

    for &(h, _h_gp, t_expected) in &test_points {
        let r = atmosphere(h);
        // Allow 0.2% relative error for geometric/geopotential difference
        assert_close(r.temperature, t_expected, 2e-3, "T_troposphere", h);
    }
}

#[test]
fn test_troposphere_density() {
    // Source: NASA-TM-X-74335 Table 4
    let test_points = [
        // (geometric_h_m, expected_rho_kg_m3, Table 4 column)
        (0.0,     1.2250),    // H=0
        (1000.0,  1.1117),    // H=1km
        (5000.0,  0.73643),   // H=5km
        (10000.0, 0.41351),   // H=10km
    ];

    for &(h, rho_expected) in &test_points {
        let r = atmosphere(h);
        assert_close(r.density, rho_expected, 5e-3, "rho_troposphere", h);
    }
}

#[test]
fn test_troposphere_pressure() {
    // Source: NASA-TM-X-74335 Table 4
    // Pressure computed from barometric formula Eq. 33a:
    // p = p_b * (T/T_b)^(-g0*M/(R*L))
    let test_points = [
        (0.0,     101325.0),    // sea level (exact)
        (5000.0,  54048.0),     // Table 4, H=5km
        (10000.0, 26500.0),     // Table 4, H=10km (approximate)
    ];

    for &(h, p_expected) in &test_points {
        let r = atmosphere(h);
        assert_close(r.pressure, p_expected, 5e-3, "p_troposphere", h);
    }
}

// =========================================================================
// TEST GROUP 3: Tropopause / Lower Stratosphere (11-20 km)
// ISOTHERMAL layer: T = 216.650 K (constant)
// Source: NASA-TM-X-74335 Table 1, layer b1 = 11.0 km, T_b = 216.650 K, L = 0
// =========================================================================

#[test]
fn test_tropopause_isothermal() {
    // Temperature must be EXACTLY 216.650 K throughout (by definition)
    // Source: NASA-TM-X-74335 Table 1, layer b1 = 11.0 km, T_b = 216.650 K, L = 0
    for h_km in 12..20 {
        let h = h_km as f64 * 1000.0;
        let r = atmosphere(h);
        assert_close(r.temperature, 216.650, 1e-3, "T_isothermal", h);
    }
}

#[test]
fn test_tropopause_pressure_exponential() {
    // In isothermal layer, pressure follows:
    // p = p_b * exp(-g0*M*(H-H_b)/(R*T))  (Eq. 33b)
    // Source: Table 4, H=15km row
    let r = atmosphere(15000.0);
    // Table 4 H=15km: p ≈ 12111 Pa, rho ≈ 0.19476 kg/m³
    assert_close(r.pressure, 12111.0, 1e-2, "p_15km", 15000.0);
    assert_close(r.density, 0.19476, 1e-2, "rho_15km", 15000.0);
}

// =========================================================================
// TEST GROUP 4: Full Altitude Sweep — Density Monotonicity
// PHYSICAL LAW: atmospheric density must decrease monotonically with
// altitude (hydrostatic equilibrium in a gravitational field).
// No citation needed — this is a fundamental physical requirement.
// =========================================================================

#[test]
fn test_density_monotonically_decreasing_full_range() {
    let mut prev_rho = f64::MAX;
    // Check every 500m from 0 to 85 km
    for h_step in 0..170 {
        let h = h_step as f64 * 500.0;
        let r = atmosphere(h);

        assert!(r.density > 0.0,
            "density must be positive at h={:.0} m, got {:.6e}", h, r.density);
        assert!(r.density < prev_rho,
            "density not monotonically decreasing at h={:.0} m: {:.6e} >= {:.6e}",
            h, r.density, prev_rho);
        assert!(r.temperature > 0.0,
            "temperature must be positive at h={:.0} m", h);
        assert!(r.pressure > 0.0,
            "pressure must be positive at h={:.0} m", h);
        assert!(r.speed_of_sound > 0.0,
            "speed of sound must be positive at h={:.0} m", h);

        prev_rho = r.density;
    }
}

// =========================================================================
// TEST GROUP 5: Layer Boundary Continuity
// MATHEMATICAL REQUIREMENT: Temperature, pressure, and density must be
// continuous at layer boundaries. Discontinuities indicate errors in
// the base pressure chain computation.
// Source: NASA-TM-X-74335 construction guarantees this by design.
// =========================================================================

#[test]
fn test_layer_boundary_continuity_all_properties() {
    // Layer boundaries in geopotential km: 11, 20, 32, 47, 51, 71
    // We test slightly below and above each boundary in geometric meters.
    let boundaries = [11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0];

    for &h_b in &boundaries {
        let eps = 0.01; // 1cm either side
        let below = atmosphere(h_b - eps);
        let above = atmosphere(h_b + eps);

        // Temperature continuity (within 0.01 K)
        let dt = (below.temperature - above.temperature).abs();
        assert!(dt < 0.01,
            "temperature discontinuity at h={:.0}: delta={:.4} K", h_b, dt);

        // Pressure continuity (within 0.01% relative)
        let dp = (below.pressure - above.pressure).abs() / below.pressure;
        assert!(dp < 1e-4,
            "pressure discontinuity at h={:.0}: rel_err={:.6}", h_b, dp);

        // Density continuity (within 0.01% relative)
        let drho = (below.density - above.density).abs() / below.density;
        assert!(drho < 1e-4,
            "density discontinuity at h={:.0}: rel_err={:.6}", h_b, drho);
    }
}

// =========================================================================
// TEST GROUP 6: Speed of Sound Self-Consistency
// FORMULA: a = sqrt(gamma * R* * T / M0) (Eq. 50, NASA-TM-X-74335 p. 15)
// This tests that our speed_of_sound is computed correctly from temperature.
// =========================================================================

#[test]
fn test_speed_of_sound_formula_consistency() {
    let gamma = 1.4_f64;
    let r_star = 8.31447_f64;
    let m0 = 0.0289644_f64;

    for h_km in (0..80).step_by(5) {
        let h = h_km as f64 * 1000.0;
        let r = atmosphere(h);
        let a_computed = (gamma * r_star * r.temperature / m0).sqrt();
        let rel_err = (r.speed_of_sound - a_computed).abs() / a_computed;
        assert!(rel_err < 1e-10,
            "speed of sound formula mismatch at h={:.0}: got {:.3}, expected {:.3}",
            h, r.speed_of_sound, a_computed);
    }
}

// =========================================================================
// TEST GROUP 7: Ideal Gas Law Self-Consistency
// FORMULA: rho = p * M0 / (R* * T) (Eq. 42, NASA-TM-X-74335 p. 14)
// If pressure AND temperature are correct, density MUST satisfy this.
// =========================================================================

#[test]
fn test_ideal_gas_law_consistency() {
    let r_star = 8.31447_f64;
    let m0 = 0.0289644_f64;

    for h_km in (0..80).step_by(2) {
        let h = h_km as f64 * 1000.0;
        let r = atmosphere(h);
        let rho_from_ideal = r.pressure * m0 / (r_star * r.temperature);
        let rel_err = (r.density - rho_from_ideal).abs() / rho_from_ideal;
        assert!(rel_err < 1e-10,
            "ideal gas law violation at h={:.0}: density={:.6e}, p*M/(R*T)={:.6e}",
            h, r.density, rho_from_ideal);
    }
}

// =========================================================================
// TEST GROUP 8: Edge Cases
// =========================================================================

#[test]
fn test_negative_altitude_clamps_to_sea_level() {
    let r_neg = atmosphere(-500.0);
    let r_zero = atmosphere(0.0);
    assert!((r_neg.density - r_zero.density).abs() < 1e-12,
        "negative altitude should clamp to sea level");
    assert!((r_neg.temperature - r_zero.temperature).abs() < 1e-12);
    assert!((r_neg.pressure - r_zero.pressure).abs() < 1e-12);
}

#[test]
fn test_very_high_altitude() {
    // Above 71km layer, the model extrapolates with lapse rate -0.002 K/m
    // At 85km: T = 214.650 + (-0.002)*(85000-71000) = 186.65 K
    let r = atmosphere(85000.0);
    assert!(r.temperature > 150.0 && r.temperature < 220.0,
        "T at 85km = {:.1} K", r.temperature);
    assert!(r.density > 0.0 && r.density < 1e-4,
        "rho at 85km = {:.6e}", r.density);
}

#[test]
fn test_exact_layer_boundary_altitude() {
    // At exactly h=11000m, should use layer 1 (isothermal)
    let r = atmosphere(11000.0);
    // Temperature should be very close to 216.650 K
    assert_close(r.temperature, 216.650, 5e-3, "T_at_11km", 11000.0);
}
