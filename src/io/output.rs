use crate::dynamics::State;
use std::io::Write;

/// Write a trajectory to CSV format (6-DOF: 15 columns).
pub fn write_trajectory_csv<W: Write>(
    writer: &mut W,
    trajectory: &[State],
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(writer, "time,x,y,z,vx,vy,vz,qw,qx,qy,qz,wx,wy,wz,mass")?;
    for s in trajectory {
        let q = s.quat.into_inner();
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            s.time,
            s.pos.x, s.pos.y, s.pos.z,
            s.vel.x, s.vel.y, s.vel.z,
            q.w, q.i, q.j, q.k,
            s.omega.x, s.omega.y, s.omega.z,
            s.mass
        )?;
    }
    Ok(())
}
