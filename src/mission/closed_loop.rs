// Closed-loop simulation: trajectory library selection + SCvx tracking.
//
// Simulates the full online guidance pipeline:
// 1. Select nearest trajectory from the library
// 2. Warm-start SCvx from the selection
// 3. Run 2-3 SCvx iterations per guidance cycle

use nalgebra::Vector3;
use crate::aero::AeroTable;
use crate::dynamics::{Control, State, G0_EARTH};
use crate::mission::simulate::{SimulationConfig, SimulationResult};
use crate::robust::Scenario;
use crate::vehicle::VehicleParams;

/// A pre-computed reference trajectory (from offline optimization).
#[derive(Debug, Clone)]
pub struct ReferenceTrajectory {
    pub id: usize,
    pub states: Vec<State>,
    pub controls: Vec<Control>,
    pub dt: f64,
}

impl ReferenceTrajectory {
    /// Get the control at a given time by interpolating the reference.
    pub fn control_at_time(&self, time: f64) -> Control {
        if self.controls.is_empty() {
            return Control::zero();
        }
        let idx = ((time / self.dt) as usize).min(self.controls.len() - 1);
        self.controls[idx].clone()
    }

    /// Get the reference state at a given time.
    pub fn state_at_time(&self, time: f64) -> &State {
        if self.states.is_empty() {
            panic!("Empty reference trajectory");
        }
        let idx = ((time / self.dt) as usize).min(self.states.len() - 1);
        &self.states[idx]
    }

    /// Distance metric: how far a state is from this trajectory.
    pub fn distance_to(&self, state: &State) -> f64 {
        self.states
            .iter()
            .map(|ref_state| {
                let pos_err = (state.pos - ref_state.pos).norm();
                let vel_err = (state.vel - ref_state.vel).norm();
                pos_err + 10.0 * vel_err
            })
            .fold(f64::MAX, f64::min)
    }
}

/// Trajectory library: a set of pre-computed trajectories for different conditions.
#[derive(Debug, Clone)]
pub struct TrajectoryLibrary {
    pub trajectories: Vec<ReferenceTrajectory>,
}

impl TrajectoryLibrary {
    pub fn new() -> Self {
        Self {
            trajectories: Vec::new(),
        }
    }

    pub fn add(&mut self, traj: ReferenceTrajectory) {
        self.trajectories.push(traj);
    }

    /// Select the best trajectory from the library for a given state.
    pub fn select(&self, state: &State) -> Option<&ReferenceTrajectory> {
        self.trajectories
            .iter()
            .min_by(|a, b| {
                a.distance_to(state)
                    .partial_cmp(&b.distance_to(state))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Closed-loop guidance controller using trajectory library + tracking.
///
/// This implements the online phase of the hybrid architecture:
/// - Select nearest reference trajectory from library
/// - Track it using proportional feedback (placeholder for SCvx)
#[derive(Debug, Clone)]
pub struct ClosedLoopController {
    pub library: TrajectoryLibrary,
    pub vehicle: VehicleParams,
    pub tracking_gain_pos: f64,
    pub tracking_gain_vel: f64,
}

impl crate::mission::simulate::Controller for ClosedLoopController {
    fn compute(&self, state: &State, time: f64) -> Control {
        // 1. Select best reference trajectory
        let ref_traj = match self.library.select(state) {
            Some(t) => t,
            None => return gravity_turn_fallback(state, &self.vehicle),
        };

        // 2. Get reference state and control at current time
        let ref_state = ref_traj.state_at_time(time);
        let ref_control = ref_traj.control_at_time(time);

        // 3. Tracking correction (proportional feedback)
        // This is a placeholder for full SCvx re-solve
        let pos_err = ref_state.pos - state.pos;
        let vel_err = ref_state.vel - state.vel;

        let correction = self.tracking_gain_pos * pos_err + self.tracking_gain_vel * vel_err;
        // Estimate reference thrust magnitude from throttle
        let ref_thrust_mag = self.vehicle.thrust_min
            + ref_control.throttle * (self.vehicle.thrust_max - self.vehicle.thrust_min);
        let corrected_mag = ref_thrust_mag + correction.norm() * state.mass;

        // Clamp to vehicle limits
        if corrected_mag < 1e-6 {
            return ref_control;
        }
        let clamped = corrected_mag.clamp(self.vehicle.thrust_min, self.vehicle.thrust_max);
        let throttle = (clamped - self.vehicle.thrust_min)
            / (self.vehicle.thrust_max - self.vehicle.thrust_min);
        Control::new(throttle.clamp(0.0, 1.0), ref_control.gimbal_y, ref_control.gimbal_z)
    }
}

/// Fallback controller when no reference trajectory is available.
fn gravity_turn_fallback(state: &State, vehicle: &VehicleParams) -> Control {
    let vel = state.vel;
    let a_desired = -1.5 * vel + Vector3::new(0.0, 0.0, G0_EARTH);
    let f_desired = a_desired * state.mass;
    let f_mag = f_desired.norm();
    let f_clamped = f_mag.clamp(vehicle.thrust_min, vehicle.thrust_max);

    if f_mag < 1e-6 {
        Control::zero()
    } else {
        let throttle = (f_clamped - vehicle.thrust_min)
            / (vehicle.thrust_max - vehicle.thrust_min);
        Control::new(throttle.clamp(0.0, 1.0), 0.0, 0.0)
    }
}

/// Run a closed-loop simulation with trajectory library.
pub fn closed_loop_simulate(
    initial_state: &State,
    vehicle: &VehicleParams,
    aero: &AeroTable,
    scenario: &Scenario,
    library: &TrajectoryLibrary,
    config: &SimulationConfig,
) -> SimulationResult {
    let controller = ClosedLoopController {
        library: library.clone(),
        vehicle: vehicle.clone(),
        tracking_gain_pos: 0.1,
        tracking_gain_vel: 1.0,
    };

    crate::mission::simulate::forward_simulate(
        initial_state,
        vehicle,
        aero,
        scenario,
        &controller,
        config,
    )
}

/// Generate a simple reference trajectory by forward-simulating
/// with a gravity-turn controller under nominal conditions.
pub fn generate_reference_trajectory(
    initial_state: &State,
    vehicle: &VehicleParams,
    aero: &AeroTable,
    dt: f64,
) -> ReferenceTrajectory {
    let nominal = Scenario::nominal(0);
    let controller = crate::mission::simulate::GravityTurnController {
        target_pos: Vector3::new(0.0, 0.0, 0.0),
        vehicle: vehicle.clone(),
        gain_vel: 1.5,
        gain_pos: 0.3,
    };

    let config = SimulationConfig {
        dt,
        max_time: 300.0,
        ground_altitude: 0.0,
        max_g_load_abort: 15.0,
        max_dynamic_pressure_abort: 200_000.0,
    };

    let result = crate::mission::simulate::forward_simulate(
        initial_state,
        vehicle,
        aero,
        &nominal,
        &controller,
        &config,
    );

    ReferenceTrajectory {
        id: 0,
        states: result.trajectory,
        controls: result.controls,
        dt,
    }
}
