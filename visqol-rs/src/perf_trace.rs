//! Lightweight wall-clock instrumentation, gated on `VISQOL_PERF_TRACE=1`.
//!
//! Toggled via env var so it stays compiled in (cheap atomic load on the
//! enabled-check fast path) but stays silent in normal use. Records cumulative
//! nanoseconds and call counts per named span, prints a summary on `dump()`.
//!
//! Usage:
//!   VISQOL_PERF_TRACE=1 cargo run --release ...
//!   {
//!       let _t = perf_trace::span("phase_name");
//!       // work
//!   } // recorded on drop
//!   perf_trace::dump();  // prints summary to stderr

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::sync::Mutex;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: OnceLock<()> = OnceLock::new();

struct Slot {
    name: &'static str,
    nanos: AtomicU64,
    calls: AtomicU64,
}

static SLOTS: Mutex<Vec<&'static Slot>> = Mutex::new(Vec::new());

fn ensure_init() {
    INIT.get_or_init(|| {
        let on = std::env::var("VISQOL_PERF_TRACE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        ENABLED.store(on, Ordering::Relaxed);
    });
}

#[inline]
pub fn enabled() -> bool {
    ensure_init();
    ENABLED.load(Ordering::Relaxed)
}

fn slot_for(name: &'static str) -> &'static Slot {
    let mut guard = SLOTS.lock().unwrap();
    if let Some(s) = guard.iter().find(|s| s.name == name) {
        return *s;
    }
    let leaked: &'static Slot = Box::leak(Box::new(Slot {
        name,
        nanos: AtomicU64::new(0),
        calls: AtomicU64::new(0),
    }));
    guard.push(leaked);
    leaked
}

pub struct Span {
    slot: Option<&'static Slot>,
    start: Instant,
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            let elapsed = self.start.elapsed().as_nanos() as u64;
            slot.nanos.fetch_add(elapsed, Ordering::Relaxed);
            slot.calls.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[inline]
pub fn span(name: &'static str) -> Span {
    if enabled() {
        Span {
            slot: Some(slot_for(name)),
            start: Instant::now(),
        }
    } else {
        Span {
            slot: None,
            start: Instant::now(),
        }
    }
}

/// Print a summary of all recorded spans to stderr.
pub fn dump() {
    if !enabled() {
        return;
    }
    let guard = SLOTS.lock().unwrap();
    let mut entries: Vec<(&'static str, u64, u64)> = guard
        .iter()
        .map(|s| {
            (
                s.name,
                s.nanos.load(Ordering::Relaxed),
                s.calls.load(Ordering::Relaxed),
            )
        })
        .collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = entries.iter().map(|e| e.1).sum();
    eprintln!("\n=== visqol perf trace ===");
    eprintln!("{:<48}  {:>12}  {:>10}  {:>8}  {:>6}", "span", "ms", "calls", "us/call", "%");
    for (name, nanos, calls) in entries {
        let ms = nanos as f64 / 1_000_000.0;
        let pct = if total > 0 { 100.0 * nanos as f64 / total as f64 } else { 0.0 };
        let us_per = if calls > 0 {
            (nanos as f64 / 1000.0) / calls as f64
        } else {
            0.0
        };
        eprintln!("{:<48}  {:>12.3}  {:>10}  {:>8.2}  {:>6.2}", name, ms, calls, us_per, pct);
    }
    eprintln!("{:<48}  {:>12.3}", "TOTAL (sum, may double-count)", total as f64 / 1_000_000.0);
}

/// Reset all counters. Useful between iterations.
pub fn reset() {
    if !enabled() {
        return;
    }
    let guard = SLOTS.lock().unwrap();
    for s in guard.iter() {
        s.nanos.store(0, Ordering::Relaxed);
        s.calls.store(0, Ordering::Relaxed);
    }
}
