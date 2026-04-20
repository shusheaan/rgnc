//! Build script for rgnc.
//!
//! When the `gurobi` feature is enabled and `GUROBI_HOME` is set,
//! generates Rust FFI bindings from the Gurobi C header via `bindgen`.
//! If `GUROBI_HOME` is not set, writes empty bindings (compile-only mode).
//!
//! Adapted from the `star` project's build.rs.

extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir).join("gurobi_bindings.rs");

    // Only generate Gurobi bindings if the feature is enabled
    if !cfg!(feature = "gurobi") {
        std::fs::write(&out_path, "// gurobi feature disabled\n").unwrap();
        return;
    }

    let gurobi_home = match env::var("GUROBI_HOME") {
        Ok(val) => val,
        Err(_) => {
            println!(
                "cargo:warning=GUROBI_HOME not set; writing empty bindings (compile-only mode)"
            );
            std::fs::write(
                &out_path,
                "// GUROBI_HOME not set — empty bindings for compile-only mode\n",
            )
            .unwrap();
            return;
        }
    };

    // Link against Gurobi shared library
    // Adjust the library name to match your Gurobi version:
    //   gurobi130 for Gurobi 13.0.x
    //   gurobi120 for Gurobi 12.0.x
    //   etc.
    println!("cargo:rustc-link-lib=gurobi130");

    let include_path = format!("{}/include", gurobi_home);
    println!("cargo:warning=Gurobi include path = {}", include_path);

    let bindings = bindgen::Builder::default()
        .header(format!("{}/include/gurobi_c.h", gurobi_home))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate Gurobi bindings");

    bindings
        .write_to_file(&out_path)
        .expect("Failed to write Gurobi bindings");

    println!(
        "cargo:warning=Gurobi bindings written to {}",
        out_path.display()
    );
}
