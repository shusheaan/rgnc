use rgnc::vehicle::VehicleParams;

fn main() {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config/acikemese2007.yaml".to_string());

    println!("RGNC - Robust Reentry Trajectory Optimization");
    println!("Loading config: {}", config_path);

    match VehicleParams::from_yaml(&config_path) {
        Ok(params) => {
            println!("Vehicle loaded:");
            println!("  Dry mass:    {:.1} kg", params.dry_mass);
            println!("  Fuel mass:   {:.1} kg", params.fuel_mass);
            println!("  Total mass:  {:.1} kg", params.total_mass());
            println!("  Thrust:      {:.1} - {:.1} N", params.thrust_min, params.thrust_max);
            println!("  Isp:         {:.1} s", params.isp);
        }
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            std::process::exit(1);
        }
    }
}
