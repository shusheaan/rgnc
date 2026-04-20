//! =========================================================================
//! CSV Trajectory Output — Verification Tests
//! =========================================================================
//!
//! WHAT THIS TESTS:
//!   The `write_trajectory_csv` function in `src/io/output.rs` serializes
//!   a Vec<State> into CSV format with columns:
//!     time, x, y, z, vx, vy, vz, mass
//!
//! WHY THIS MATTERS:
//!   CSV output is the primary way trajectory data is exported for:
//!   - Python visualization scripts (matplotlib)
//!   - External validation against paper figures
//!   - Monte Carlo result aggregation
//!   If the column order or formatting is wrong, all downstream analysis
//!   will silently produce incorrect results.
//!
//! HOW TO VERIFY:
//!   The tests construct known State vectors and check the CSV output
//!   string character-by-character. No external data needed.
//! =========================================================================

use nalgebra::{UnitQuaternion, Vector3};
use rgnc::dynamics::eom::State;
use rgnc::io::output::write_trajectory_csv;

#[test]
fn test_csv_output_format() {
    // WHAT: Verify CSV header and data formatting for a 2-point trajectory.
    // WHY: Column order must match "time,x,y,z,vx,vy,vz,mass" exactly,
    //       otherwise plot_trajectory.py will assign wrong axes.
    // VERIFY: Read the output string — header + 2 data lines, correct values.
    let trajectory = vec![
        State::new(
            Vector3::new(1.0, 2.0, 3.0),   // pos = (1, 2, 3)
            Vector3::new(4.0, 5.0, 6.0),   // vel = (4, 5, 6)
            UnitQuaternion::identity(),
            Vector3::new(0.1, 0.2, 0.3),
            100.0,                           // mass = 100 kg
            0.0,                             // time = 0 s
        ),
        State::new(
            Vector3::new(7.0, 8.0, 9.0),   // pos = (7, 8, 9)
            Vector3::new(10.0, 11.0, 12.0), // vel = (10, 11, 12)
            UnitQuaternion::identity(),
            Vector3::zeros(),
            99.0,                            // mass = 99 kg (1 kg fuel burned)
            1.0,                             // time = 1 s
        ),
    ];

    let mut buf = Vec::new();
    write_trajectory_csv(&mut buf, &trajectory).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    // Header must be exactly this string (defines column semantics)
    assert!(csv.starts_with("time,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,mass\n"),
        "CSV header incorrect: got {:?}", csv.lines().next());

    // Must have exactly 3 lines: 1 header + 2 data rows
    assert_eq!(csv.lines().count(), 3,
        "Expected 3 lines (header + 2 data), got {}", csv.lines().count());

    // First data line: time=0, pos=(1,2,3), vel=(4,5,6), quat=identity(1,0,0,0), omega=(0.1,0.2,0.3), mass=100
    let line1 = csv.lines().nth(1).unwrap();
    assert!(line1.starts_with("0,1,2,3,4,5,6,1,0,0,0,"),
        "First data line incorrect: got {:?}", line1);
}

#[test]
fn test_csv_output_empty_trajectory() {
    // WHAT: Verify that an empty trajectory produces header-only CSV.
    // WHY: Edge case — Monte Carlo runs that abort immediately should
    //       still produce valid CSV files that downstream tools can parse
    //       without crashing on empty data.
    let trajectory: Vec<State> = vec![];
    let mut buf = Vec::new();
    write_trajectory_csv(&mut buf, &trajectory).unwrap();
    let csv = String::from_utf8(buf).unwrap();

    // Should have exactly 1 line (header only, no data)
    assert_eq!(csv.lines().count(), 1,
        "Empty trajectory should produce header-only CSV");
    assert!(csv.contains("time,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,mass"),
        "Header missing from empty CSV");
}
