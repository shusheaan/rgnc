use crate::vehicle::VehicleParams;

/// Load mission configuration from a YAML file.
pub fn load_vehicle_config(path: &str) -> Result<VehicleParams, Box<dyn std::error::Error>> {
    VehicleParams::from_yaml(path)
}
