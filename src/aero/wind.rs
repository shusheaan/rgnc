use nalgebra::Vector3;
use crate::robust::Scenario;

/// Wind profile: wind velocity as a function of altitude.
#[derive(Debug, Clone)]
pub struct WindProfile {
    pub altitudes: Vec<f64>,
    pub wind_vectors: Vec<Vector3<f64>>,
}

impl WindProfile {
    /// No-wind (calm) profile.
    pub fn calm() -> Self {
        Self {
            altitudes: vec![0.0, 100_000.0],
            wind_vectors: vec![Vector3::zeros(), Vector3::zeros()],
        }
    }

    /// Constant wind at all altitudes.
    pub fn constant(wind: Vector3<f64>) -> Self {
        Self {
            altitudes: vec![0.0, 100_000.0],
            wind_vectors: vec![wind, wind],
        }
    }

    /// Construct a WindProfile from a scenario's wind data.
    pub fn from_scenario(scenario: &Scenario) -> Self {
        let (altitudes, wind_vectors): (Vec<f64>, Vec<Vector3<f64>>) =
            scenario.wind_profile.iter().cloned().unzip();
        Self { altitudes, wind_vectors }
    }

    /// Interpolate wind at a given altitude.
    pub fn at_altitude(&self, h: f64) -> Vector3<f64> {
        if self.altitudes.is_empty() {
            return Vector3::zeros();
        }
        if h <= self.altitudes[0] {
            return self.wind_vectors[0];
        }
        if h >= *self.altitudes.last().unwrap() {
            return *self.wind_vectors.last().unwrap();
        }

        for i in 0..self.altitudes.len() - 1 {
            if h < self.altitudes[i + 1] {
                let t = (h - self.altitudes[i]) / (self.altitudes[i + 1] - self.altitudes[i]);
                return self.wind_vectors[i] * (1.0 - t) + self.wind_vectors[i + 1] * t;
            }
        }
        *self.wind_vectors.last().unwrap()
    }
}
