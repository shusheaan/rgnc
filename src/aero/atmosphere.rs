/// US Standard Atmosphere 1976 model.
///
/// Returns density, temperature, pressure, and speed of sound
/// for a given geometric altitude in meters.

const G0: f64 = 9.80665;
const M_AIR: f64 = 0.0289644;
const R_GAS: f64 = 8.31447;
const GAMMA: f64 = 1.4;

/// Effective Earth radius for geopotential height conversion (NASA-TM-X-74335 p. 3).
const R0_EARTH: f64 = 6_356_766.0;

/// Convert geometric altitude h (meters) to geopotential height H (meters).
///
/// H = R0 * h / (R0 + h)
pub fn geometric_to_geopotential(h: f64) -> f64 {
    R0_EARTH * h / (R0_EARTH + h)
}

/// Layer definitions: (h_base_m, t_base_K, lapse_rate_K_per_m)
const LAYER_DEFS: [(f64, f64, f64); 7] = [
    (0.0,     288.150, -0.0065),
    (11000.0, 216.650,  0.0),
    (20000.0, 216.650,  0.001),
    (32000.0, 228.650,  0.0028),
    (47000.0, 270.650,  0.0),
    (51000.0, 270.650, -0.0028),
    (71000.0, 214.650, -0.002),
];

/// Full atmosphere properties at a given altitude.
pub struct AtmosphereResult {
    pub density: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub speed_of_sound: f64,
}

/// Compute pressure at the top of a layer, given base conditions.
fn layer_pressure(p_base: f64, t_base: f64, lapse: f64, dh: f64) -> f64 {
    if lapse.abs() < 1e-10 {
        p_base * (-G0 * M_AIR / (R_GAS * t_base) * dh).exp()
    } else {
        let t_top = t_base + lapse * dh;
        p_base * (t_top / t_base).powf(-G0 * M_AIR / (R_GAS * lapse))
    }
}

/// Compute atmosphere properties at a given geometric altitude (meters).
/// Valid from 0 to ~86 km. Self-consistent pressure chain from sea level.
pub fn atmosphere(h: f64) -> AtmosphereResult {
    let h = geometric_to_geopotential(h.max(0.0));

    // Chain-compute base pressures from sea level (self-consistent)
    let mut p_base = 101325.0; // sea level pressure, Pa
    let mut layer_idx = 0;

    for i in 0..LAYER_DEFS.len() - 1 {
        let (h_b, t_b, lapse) = LAYER_DEFS[i];
        let (h_next, _, _) = LAYER_DEFS[i + 1];
        if h >= h_next {
            p_base = layer_pressure(p_base, t_b, lapse, h_next - h_b);
            layer_idx = i + 1;
        } else {
            break;
        }
    }

    let (h_b, t_b, lapse) = LAYER_DEFS[layer_idx];
    let dh = h - h_b;

    let (temperature, pressure) = if lapse.abs() < 1e-10 {
        let t = t_b;
        let p = p_base * (-G0 * M_AIR / (R_GAS * t) * dh).exp();
        (t, p)
    } else {
        let t = t_b + lapse * dh;
        let p = p_base * (t / t_b).powf(-G0 * M_AIR / (R_GAS * lapse));
        (t, p)
    };

    let density = pressure * M_AIR / (R_GAS * temperature);
    let speed_of_sound = (GAMMA * R_GAS * temperature / M_AIR).sqrt();

    AtmosphereResult {
        density,
        temperature,
        pressure,
        speed_of_sound,
    }
}
