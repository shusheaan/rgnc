/// Trust region management for the SCvx loop.
///
/// Adjusts the trust region radius based on the agreement between
/// the linearized model prediction and the actual nonlinear cost.
///
/// Classical trust region update rule:
///   ρ = (J_actual - J_ref) / (J_predicted - J_ref)
///   if ρ > 0.75: expand (r *= 2)    — good model agreement
///   if ρ < 0.25: shrink (r *= 0.5)  — poor agreement
///   if ρ < 0:    reject step, shrink more

/// Trust region state tracker.
#[derive(Debug, Clone)]
pub struct TrustRegion {
    pub radius: f64,
    pub min_radius: f64,
    pub max_radius: f64,
}

impl TrustRegion {
    pub fn new(initial_radius: f64) -> Self {
        Self {
            radius: initial_radius,
            min_radius: 1e-8,
            max_radius: 100.0,
        }
    }

    pub fn shrink(&mut self, factor: f64) {
        self.radius = (self.radius * factor).max(self.min_radius);
    }

    pub fn expand(&mut self, factor: f64) {
        self.radius = (self.radius * factor).min(self.max_radius);
    }
}
