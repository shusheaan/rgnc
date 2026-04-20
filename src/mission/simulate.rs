// Forward simulation with controller.
//
// Integrates the equations of motion forward in time with a
// feedback controller providing thrust commands at each step.

use nalgebra::Vector3;
use crate::aero::{AeroTable, WindProfile};
use crate::dynamics::{Control, DynamicsParams, State, derivatives_6dof};
use crate::dynamics::integrator::rk4_step;
use crate::robust::Scenario;
use crate::vehicle::VehicleParams;

/// Result of a forward simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub trajectory: Vec<State>,
    pub controls: Vec<Control>,
    pub landing_error: f64,
    pub final_speed: f64,
    pub fuel_used: f64,
    pub max_g_load: f64,
    pub max_dynamic_pressure: f64,
    pub landed: bool,
    pub aborted: bool,
    pub abort_reason: Option<String>,
}

impl SimulationResult {
    /// Whether the simulation is a successful landing.
    pub fn is_success(&self, max_landing_error: f64, max_landing_speed: f64) -> bool {
        self.landed && !self.aborted
            && self.landing_error < max_landing_error
            && self.final_speed < max_landing_speed
    }
}

/// Trait for guidance controllers.
pub trait Controller {
    /// Compute thrust command given current state and time.
    fn compute(&self, state: &State, time: f64) -> Control;
}

/// Simple gravity-turn controller for landing.
///
/// Points thrust opposite to velocity with magnitude proportional
/// to the remaining velocity, simulating a basic proportional-navigation
/// approach to zero velocity at the target.
#[derive(Debug, Clone)]
pub struct GravityTurnController {
    pub target_pos: Vector3<f64>,
    pub vehicle: VehicleParams,
    pub gain_vel: f64,
    pub gain_pos: f64,
}

impl Controller for GravityTurnController {
    fn compute(&self, state: &State, _time: f64) -> Control {
        let pos_err = self.target_pos - state.pos;
        let vel = state.vel;

        // Desired acceleration: cancel velocity + steer toward target + cancel gravity
        let a_desired = -self.gain_vel * vel + self.gain_pos * pos_err
            + Vector3::new(0.0, 0.0, crate::dynamics::G0_EARTH);

        let f_desired = a_desired * state.mass;
        let f_mag = f_desired.norm();

        // Clamp to vehicle thrust limits
        let f_clamped = f_mag.clamp(self.vehicle.thrust_min, self.vehicle.thrust_max);

        if f_mag < 1e-6 {
            return Control::zero();
        }

        let throttle = (f_clamped - self.vehicle.thrust_min)
            / (self.vehicle.thrust_max - self.vehicle.thrust_min);
        Control::new(throttle.clamp(0.0, 1.0), 0.0, 0.0)
    }
}

/// Ballistic (no thrust) controller.
#[derive(Debug, Clone)]
pub struct BallisticController;

impl Controller for BallisticController {
    fn compute(&self, _state: &State, _time: f64) -> Control {
        Control::zero()
    }
}

/// Configuration for the forward simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub dt: f64,
    pub max_time: f64,
    pub ground_altitude: f64,
    pub max_g_load_abort: f64,
    pub max_dynamic_pressure_abort: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            dt: 0.1,
            max_time: 600.0,
            ground_altitude: 0.0,
            max_g_load_abort: 15.0,
            max_dynamic_pressure_abort: 100_000.0,
        }
    }
}

/// Run a forward simulation with the given controller and scenario.
pub fn forward_simulate(
    initial_state: &State,
    vehicle: &VehicleParams,
    aero: &AeroTable,
    scenario: &Scenario,
    controller: &dyn Controller,
    config: &SimulationConfig,
) -> SimulationResult {
    let wind = WindProfile::from_scenario(scenario);
    let initial_mass = initial_state.mass;

    let params = DynamicsParams {
        vehicle,
        aero,
        wind: &wind,
        density_factor: scenario.density_factor,
        cd_factor: scenario.cd_factor,
        thrust_bias: scenario.thrust_bias,
    };

    let mut state = initial_state.clone();
    let mut trajectory = vec![state.clone()];
    let mut controls = Vec::new();
    let mut max_g_load: f64 = 0.0;
    let mut max_q_dyn: f64 = 0.0;
    let mut aborted = false;
    let mut abort_reason = None;

    let g0 = crate::dynamics::G0_EARTH;

    while state.time < config.max_time {
        // Check ground contact
        if state.pos.z <= config.ground_altitude {
            break;
        }

        // Controller command
        let control = controller.compute(&state, state.time);
        controls.push(control.clone());

        // Integrate one step
        let ctrl_ref = &control;
        let params_ref = &params;
        state = rk4_step(&state, config.dt, |s| {
            derivatives_6dof(s, ctrl_ref, params_ref)
        });

        // Track metrics
        let q_dyn = state.dynamic_pressure();
        max_q_dyn = max_q_dyn.max(q_dyn);

        // G-load: |net acceleration| / g0
        let accel = {
            let dot = derivatives_6dof(&state, &control, &params);
            dot.vel_dot.norm()
        };
        let g_load = accel / g0;
        max_g_load = max_g_load.max(g_load);

        // Abort checks
        if g_load > config.max_g_load_abort {
            aborted = true;
            abort_reason = Some(format!("G-load exceeded: {:.1}g > {:.1}g", g_load, config.max_g_load_abort));
            break;
        }
        if q_dyn > config.max_dynamic_pressure_abort {
            aborted = true;
            abort_reason = Some(format!("Dynamic pressure exceeded: {:.0} Pa > {:.0} Pa", q_dyn, config.max_dynamic_pressure_abort));
            break;
        }

        // Mass depletion check
        if state.mass <= vehicle.dry_mass {
            state.mass = vehicle.dry_mass;
            aborted = true;
            abort_reason = Some("Fuel exhausted".to_string());
            break;
        }

        trajectory.push(state.clone());
    }

    let landing_error = (state.pos - Vector3::new(0.0, 0.0, config.ground_altitude)).norm();
    let final_speed = state.vel.norm();
    let fuel_used = initial_mass - state.mass;
    let landed = state.pos.z <= config.ground_altitude + 1.0; // within 1m of ground

    SimulationResult {
        trajectory,
        controls,
        landing_error,
        final_speed,
        fuel_used,
        max_g_load,
        max_dynamic_pressure: max_q_dyn,
        landed,
        aborted,
        abort_reason,
    }
}
