use nalgebra::Vector3;
use serde::Deserialize;

/// Vehicle physical parameters loaded from YAML config.
#[derive(Debug, Clone, Deserialize)]
pub struct VehicleParams {
    pub dry_mass: f64,
    pub fuel_mass: f64,
    pub thrust_max: f64,
    pub thrust_min: f64,
    pub isp: f64,
    pub ref_area: f64,
    pub glideslope_angle: f64,
    /// Principal moments of inertia [Jx, Jy, Jz] (kg*m^2).
    #[serde(default = "default_inertia")]
    pub inertia: Vector3<f64>,
    /// Aerodynamic reference length (m).
    #[serde(default = "default_ref_length")]
    pub ref_length: f64,
    /// Center of gravity offset in body frame (m).
    #[serde(default)]
    pub cg_offset: Vector3<f64>,
    /// Center of pressure offset in body frame (m).
    #[serde(default = "default_cp_offset")]
    pub cp_offset: Vector3<f64>,
    /// Engine mount point relative to CG in body frame (m).
    #[serde(default = "default_engine_offset")]
    pub engine_offset: Vector3<f64>,
    /// Maximum gimbal angle (rad).
    #[serde(default = "default_gimbal_max")]
    pub gimbal_max: f64,
}

fn default_inertia() -> Vector3<f64> {
    Vector3::new(1000.0, 1000.0, 200.0)
}

fn default_ref_length() -> f64 {
    3.0
}

fn default_cp_offset() -> Vector3<f64> {
    Vector3::new(2.0, 0.0, 0.0)
}

fn default_engine_offset() -> Vector3<f64> {
    Vector3::new(-3.0, 0.0, 0.0)
}

fn default_gimbal_max() -> f64 {
    0.2618 // ~15 degrees
}

impl VehicleParams {
    pub fn total_mass(&self) -> f64 {
        self.dry_mass + self.fuel_mass
    }

    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let params: VehicleParams = serde_yaml::from_str(&contents)?;
        Ok(params)
    }
}
