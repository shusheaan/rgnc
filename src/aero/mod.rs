pub mod atmosphere;
pub mod coefficients;
pub mod wind;

pub use atmosphere::{atmosphere, AtmosphereResult};
pub use coefficients::AeroTable;
pub use wind::WindProfile;
