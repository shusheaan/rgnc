# Robust Reentry Trajectory Optimization — Project Plan

## Core Thesis

Scenario-based robust optimization (MIP) expands the operational weather envelope for rocket
landing compared to traditional nominal trajectory + manual margin approaches. The system
demonstrates this through three methods:

| Method | Description | Expected Success | Fuel Cost |
|--------|-------------|-----------------|-----------|
| A: Nominal | Single trajectory + SCvx | ~85% | Baseline |
| B: Worst-case | Protect ALL scenarios | ~100% | +15-20% |
| C: Chance-constrained MIP | Protect 95% of scenarios | ~97% | +5-8% |

Key insight: Method C achieves near-B reliability with much less fuel penalty.

## Architecture

```
OFFLINE (Ground, unlimited compute)          ONLINE (Vehicle, 100ms/cycle)
┌──────────────────────────────────┐        ┌──────────────────────────────┐
│ MIP: scenario-based optimization │        │ SOCP/SCvx: real-time tracking│
│  - Chance-constrained trajectory │  ───►  │  - Select trajectory from lib│
│  - Trajectory library (set cover)│        │  - Warm-start SCvx           │
│  - Safety boundary computation   │        │  - 2-3 SCvx iters per cycle  │
└──────────────────────────────────┘        └──────────────────────────────┘
```

## Current Status (105 tests passing)

### Implemented
| Component | Module | Tests |
|-----------|--------|-------|
| US Std Atmosphere 1976 (0-86km) | `src/aero/atmosphere.rs` | 13 |
| Gravity (constant + altitude) | `src/dynamics/gravity.rs` | 7 |
| Aero coefficients (bilinear interp) | `src/aero/coefficients.rs` | 9 |
| Wind profiles | `src/aero/wind.rs` | incl |
| RK4 integrator (quat normalization) | `src/dynamics/integrator.rs` | 9 |
| 6-DOF dynamics (quat, gimbal, aero) | `src/dynamics/eom.rs` | 14 |
| **2D/3D PDG SOCP** (Açıkmeşe 2007) | `src/guidance/formulation.rs` | via test_pdg |
| **Free-tf PDG** (golden section) | `src/guidance/pdg.rs` | via test_pdg |
| **SCvx iteration loop** | `src/guidance/scvx.rs` | via test_scvx |
| **SCvx SOCP subproblem** | `src/guidance/scvx_formulation.rs` | via test_scvx |
| **Linearization** (finite diff) | `src/guidance/linearize.rs` | via test_scvx |
| Forward simulation (6-DOF) | `src/mission/simulate.rs` | 8 |
| Monte Carlo (parallel, rayon) | `src/mission/montecarlo.rs` | 8 |
| Closed-loop guidance | `src/mission/closed_loop.rs` | 7 |
| Trajectory library | `src/robust/library.rs` | incl |
| Gurobi MIP (C FFI) | `src/gurobi/` | 5 |
| Vehicle params (YAML) | `src/vehicle/params.rs` | 6 |
| CSV output | `src/io/output.rs` | 2 |

### Stub / Not Yet Implemented
| Component | Module | Tier |
|-----------|--------|------|
| Chance-constrained MIP | `src/robust/chance_constr.rs` | 3 |
| Trajectory library MIP (set cover) | `src/robust/library.rs` | 3 |
| Safety boundary computation | `src/robust/safety_boundary.rs` | 3 |
| NOAA wind data loading | `src/aero/wind.rs` | 3 |

---

## Tier Roadmap

### Tier 1: Optimizer Works ✅ DONE
SOCP PDG in 2D/3D, validated against Açıkmeşe 2007.
- Trapezoidal + Euler discretization
- Lossless convexification (thrust gap relaxation)
- Free final time via golden-section search
- Glideslope SOC constraint

### Tier 2: Physics Reasonable — IN PROGRESS
6-DOF simulation + SCvx with aerodynamic drag.
- SCvx iteration loop with trust region ✅
- Finite-difference Jacobians ✅
- Forward simulation defect check ✅
- **TODO**: validate thrust params (single vs 6-engine, see Open Questions)
- **TODO**: end-to-end Tier 2 benchmark (Problem Set 3)

### Tier 3: Robust Optimization Demo
MIP-based scenario optimization, trajectory library, closed-loop Monte Carlo.
- Scenario generation framework (exists, needs MIP integration)
- Chance-constrained MIP formulation (stub)
- Trajectory library set cover (stub)
- Comparative Monte Carlo: nominal vs worst-case vs chance-constrained

---

## Open Research Questions

### Q1: Thrust Parameters (Blocks Validation)
Paper lists per-engine 3100/1400 N. Current code uses single-engine.
6-engine total (18600/8400 N) gives T/W=2.63, physically correct for hover.
**Decision**: Switch to 6-engine config for Appendix D benchmark.

### Q3: MIP Formulation — Core Novel Contribution
Three options:
- **A**: Direct scenario-based chance constraints (big-M, Calafiore 2006)
- **B**: Trajectory library as set cover / facility location
- **C**: Hybrid — MIP selects critical scenarios, SOCP robust counterpart
**Decision needed**: Which is the publication target?

### Q6: Publication Target
- Option A: "MIP trajectory library outperforms nominal" → AIAA GNC conf
- Option B: "Chance-constrained MIP provides optimal margins" → JGCD
- Option C: "Decomposition scheme for tractable robust reentry" → OR journal

### Q8: Online SCvx vs Offline-Only
Offline-only (pre-compute library + Monte Carlo) may suffice for first paper.
**Decision needed**: Online 10Hz SCvx vs offline-only?

---

## Benchmark Parameters (Açıkmeşe 2007)

```yaml
# 2D/3D PDG primary benchmark — 6-engine config
gravity: 3.7114             # Mars, constant, -z
wet_mass: 1905.0            # kg
dry_mass: 1505.0            # kg
thrust_max: 18600.0         # N (6 × 3100)
thrust_min: 8400.0          # N (6 × 1400)
isp: 225.0                  # s
alpha: 4.53e-4              # 1/(Isp * g0)
glideslope_angle_deg: 4.0
initial_position: [2000.0, 0.0, 1500.0]  # m
initial_velocity: [-75.0, 0.0, 100.0]    # m/s (vz=+100 = upward)
target: origin, zero velocity
n_timesteps: 50
```

Pass criteria: solver Solved, ‖r_N‖ < 0.01m, ‖v_N‖ < 0.01m/s,
fuel ∈ [100, 300] kg, tf ∈ [30, 120]s, no thrust in gap (0, T_min),
glideslope satisfied, altitude ≥ 0, fuel monotone with altitude sweep.

---

## References

1. Açıkmeşe & Ploen (2007) — "Convex Programming Approach to PDG for Mars Landing", JGCD 30(5)
2. Blackmore et al. (2010) — "Minimum-Landing-Error Powered-Flight Guidance", JGCD 33(4)
3. Blackmore (2016) — "Autonomous Precision Landing of Space Rockets", Bridge 46(4)
4. Szmuk & Açıkmeşe (2018) — "SCvx for 6-DoF Mars Rocket Powered Landing", AIAA 2018-0617
5. Malyuta et al. (2022) — "Convex Optimization for Trajectory Generation", IEEE CSM
6. Blackmore, Ono & Williams (2011) — "Chance-Constrained Optimal Path Planning", IEEE T-RO 27(6)
7. Ben-Tal, El Ghaoui & Nemirovski (2009) — "Robust Optimization", Princeton
8. Calafiore & Campi (2006) — "Scenario Approach to Robust Control Design", IEEE TAC 51(5)
9. Clarabel docs: https://oxfordcontrol.github.io/ClarabelDocs/
10. US Standard Atmosphere 1976: NASA-TM-X-74335

## Build & Test

```bash
cargo build
cargo test              # 105 tests
cargo test -- --nocapture
```

Gurobi optional (feature-gated). Without `GUROBI_HOME`, compiles with stubs.
