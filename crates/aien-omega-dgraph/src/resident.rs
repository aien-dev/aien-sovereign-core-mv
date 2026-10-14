//! Process-wide switches of omega's resident decode chain (OWN-30B W-L2h / W-L2i, omega 802a0101):
//! `omega_gpu_decode_graph_set_resident`, `_set_resident_grid`, `_set_resident_rows` and their getters
//! (`src/omega_gpu_decode_graph.h`). On, each layer of a one-list decode token records [o_proj + add] and
//! [gate_up, swiglu, down, combine + norm, next qkv] as one resident launch each (436 -> 197 launches per
//! token on the full model); the tokens are bit-identical to the launched list (omega's own gate, and the
//! sc full-model gate own30b-L2h-resident-20261014T0340Z).
//!
//! An explicit set here wins over omega's own environment fallback (`AIEN_OMEGA_DGRAPH_RESIDENT`,
//! `_GRID`, `_ROWS`), which omega reads only when nothing was ever set. Omega builds the resident stage
//! table at the first resident token, so set these before the first decode token.
//!
//! `is_native()` false (an omega before the row-pair kernel, or the CPU stub): [`set`] refuses an ON
//! request and [`get`] reports off.

/// Whether the resident switches are linked.
pub const fn is_native() -> bool {
    cfg!(has_omega_dgraph_resident)
}

/// The resident decode chain settings omega will use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resident {
    pub on: bool,
    /// CTAs of the resident kernel (omega default 128; the gated value).
    pub grid: u32,
    /// GEMV rows per warp at a time: 1, or 2 (row pairs, the gated value).
    pub rows: u32,
}

/// The settings of the sc full-model gate (grid 128, rows 2).
pub const GATED: Resident = Resident { on: true, grid: 128, rows: 2 };

#[cfg(has_omega_dgraph_resident)]
mod ffi {
    extern "C" {
        pub fn omega_gpu_decode_graph_set_resident(on: i32);
        pub fn omega_gpu_decode_graph_resident() -> i32;
        pub fn omega_gpu_decode_graph_set_resident_grid(grid: u32);
        pub fn omega_gpu_decode_graph_resident_grid() -> u32;
        pub fn omega_gpu_decode_graph_set_resident_rows(rows: u32);
        pub fn omega_gpu_decode_graph_resident_rows() -> u32;
    }
}

/// Sets all three switches. Off sets only the on/off switch (grid and rows are then unused).
/// Refuses rows other than 1 or 2, a zero grid, and ON in a binary without the switches.
pub fn set(r: Resident) -> Result<(), String> {
    if r.on {
        if r.grid == 0 {
            return Err("resident decode: grid 0".into());
        }
        if r.rows != 1 && r.rows != 2 {
            return Err(format!("resident decode: rows {} (1 or 2)", r.rows));
        }
    }
    #[cfg(has_omega_dgraph_resident)]
    {
        // SAFETY: plain process-wide setters on integers.
        unsafe {
            if r.on {
                ffi::omega_gpu_decode_graph_set_resident_grid(r.grid);
                ffi::omega_gpu_decode_graph_set_resident_rows(r.rows);
            }
            ffi::omega_gpu_decode_graph_set_resident(r.on as i32);
        }
        Ok(())
    }
    #[cfg(not(has_omega_dgraph_resident))]
    {
        if r.on {
            Err("resident decode: this binary has no resident decode chain linked (omega before 802a0101, or a CPU stub build)".into())
        } else {
            Ok(())
        }
    }
}

/// What omega will use (its environment fallback applies when nothing was set).
pub fn get() -> Resident {
    #[cfg(has_omega_dgraph_resident)]
    {
        // SAFETY: plain process-wide getters.
        unsafe {
            Resident {
                on: ffi::omega_gpu_decode_graph_resident() != 0,
                grid: ffi::omega_gpu_decode_graph_resident_grid(),
                rows: ffi::omega_gpu_decode_graph_resident_rows(),
            }
        }
    }
    #[cfg(not(has_omega_dgraph_resident))]
    {
        Resident { on: false, grid: 0, rows: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_refuses_bad_values_and_round_trips() {
        assert!(set(Resident { on: true, grid: 0, rows: 2 }).is_err());
        assert!(set(Resident { on: true, grid: 128, rows: 3 }).is_err());
        assert!(set(Resident { on: false, grid: 0, rows: 0 }).is_ok());
        assert!(!get().on);
        if is_native() {
            set(GATED).unwrap();
            assert_eq!(get(), GATED);
            set(Resident { on: false, ..GATED }).unwrap();
            assert!(!get().on);
        } else {
            assert!(set(GATED).is_err());
        }
    }
}
