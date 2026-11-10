//! C ABI bridge between the C++ engine and the vendored upstream Renju VCF
//! solver (`vendor/quintet`, MIT, pinned commit — see
//! `vendor/quintet/UPSTREAM.md`).
//!
//! The bridge does no search of its own: it converts the engine's board into
//! the upstream representation, calls `quintet::mate::solve` in VCF mode, and
//! reports the outcome. It exists so the C++ adapter never reimplements the
//! algorithm and so the upstream result vocabulary (proven / disproven /
//! aborted, plus the ending evidence and the winning line) survives the
//! language boundary unchanged.

use std::ffi::c_char;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use quintet::board::{Board, Player, Point, Points};
use quintet::mate::{solve_with_stats, End, SolveLimits, SolveMode, SolveResult};

/// Status codes shared with `engine/src/vcf_bridge.h`.
pub const STATUS_PROVEN: i32 = 0;
pub const STATUS_DISPROVEN: i32 = 1;
pub const STATUS_BUDGET_EXHAUSTED: i32 = 2;
pub const STATUS_INTERRUPTED: i32 = 3;
pub const STATUS_INVALID_INPUT: i32 = 4;
pub const STATUS_UNSUPPORTED: i32 = 5;

/// Ending-evidence kinds reported by the upstream `End` enum.
pub const END_NONE: i32 = 0;
pub const END_FOURS: i32 = 1;
pub const END_FORBIDDEN: i32 = 2;
pub const END_UNKNOWN: i32 = 3;

/// Longest variation the bridge reports. 64 points is far beyond the deepest
/// VCF the engine asks for (at most `2 * max_attack_plies - 1`).
pub const MAX_PATH: usize = 512;
pub const MESSAGE_LEN: usize = 192;

pub const MAX_ATTACK_PLIES: u8 = 255;

/// Upstream identity recorded in the binary, for `about`/logs.
static VERSION: &[u8] = b"quintet@8219b71be2e12e77b735a4ef1462b3dd804f7b97 (MIT)\0";

/// One VCF question. `cells` uses the engine's layout: index = y * 15 + x,
/// value 0 empty / 1 black / 2 white.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VcfBridgeQuery {
    pub cells: [u8; 225],
    pub side_to_move: u8,
    pub rule: u8,
    pub max_attack_plies: u8,
    pub _padding: u8,
    pub max_nodes: u64,
    pub time_limit_ms: u32,
}

/// The answer. `path` alternates attacker and defender moves as engine move
/// indices (y * 15 + x); it is empty when nothing was proven.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VcfBridgeResult {
    pub status: i32,
    pub end_kind: i32,
    pub end_a: i32,
    pub end_b: i32,
    pub first_move: i32,
    pub path_len: i32,
    pub path: [i32; MAX_PATH],
    pub nodes: u64,
    pub elapsed_ms: u32,
    pub _padding: u32,
    pub message: [c_char; MESSAGE_LEN],
}

impl VcfBridgeResult {
    fn empty() -> Self {
        Self {
            status: STATUS_INVALID_INPUT,
            end_kind: END_NONE,
            end_a: -1,
            end_b: -1,
            first_move: -1,
            path_len: 0,
            path: [-1; MAX_PATH],
            nodes: 0,
            elapsed_ms: 0,
            _padding: 0,
            message: [0; MESSAGE_LEN],
        }
    }

    fn message(&mut self, text: &str) {
        let bytes = text.as_bytes();
        let n = bytes.len().min(MESSAGE_LEN - 1);
        for (slot, byte) in self.message.iter_mut().zip(bytes.iter()) {
            *slot = *byte as c_char;
        }
        self.message[n] = 0;
    }
}

/// Converts an upstream point into an engine move index (y * 15 + x).
fn move_of(point: Point) -> i32 {
    i32::from(point.1) * 15 + i32::from(point.0)
}

fn run(query: &VcfBridgeQuery) -> VcfBridgeResult {
    let started = Instant::now();
    let mut result = VcfBridgeResult::empty();

    if query.rule != 2 {
        result.status = STATUS_UNSUPPORTED;
        result.message(
            "upstream quintet models Renju rules only; freestyle VCF is disabled",
        );
        result.elapsed_ms = started.elapsed().as_millis() as u32;
        return result;
    }
    if query.side_to_move != 1 && query.side_to_move != 2 {
        result.status = STATUS_INVALID_INPUT;
        result.message("side_to_move must be 1 (black) or 2 (white)");
        return result;
    }
    if query.max_attack_plies == 0 || query.max_attack_plies > MAX_ATTACK_PLIES {
        result.status = STATUS_INVALID_INPUT;
        result.message("max_attack_plies must be in 1..=255");
        return result;
    }

    let mut blacks = Vec::with_capacity(60);
    let mut whites = Vec::with_capacity(60);
    for (index, cell) in query.cells.iter().enumerate() {
        let point = Point((index % 15) as u8, (index / 15) as u8);
        match cell {
            0 => {}
            1 => blacks.push(point),
            2 => whites.push(point),
            other => {
                result.status = STATUS_INVALID_INPUT;
                result.message(&format!(
                    "invalid cell value {other} at x={} y={}",
                    index % 15,
                    index / 15
                ));
                return result;
            }
        }
    }

    let board = Board::from_stones(&Points(blacks), &Points(whites));
    let attacker = if query.side_to_move == 1 {
        Player::Black
    } else {
        Player::White
    };
    let limit = query.max_attack_plies;
    let limits = if query.max_nodes == 0 {
        SolveLimits::new(limit)
    } else {
        SolveLimits::new(limit).with_max_nodes(query.max_nodes)
    };

    let (outcome, stats) = solve_with_stats(SolveMode::VCFDFS, &board, attacker, limits);
    result.nodes = stats.nodes;
    result.elapsed_ms = started.elapsed().as_millis() as u32;

    match outcome {
        SolveResult::Proven(mate) => {
            result.status = STATUS_PROVEN;
            result.first_move = mate.path.first().map(|p| move_of(*p)).unwrap_or(-1);
            let len = mate.path.len().min(MAX_PATH);
            for (slot, point) in result.path.iter_mut().zip(mate.path.iter()) {
                *slot = move_of(*point);
            }
            result.path_len = len as i32;
            if mate.path.len() > MAX_PATH {
                result.message("variation truncated at 512 points");
            }
            match mate.end {
                End::Fours(a, b) => {
                    result.end_kind = END_FOURS;
                    result.end_a = move_of(a);
                    result.end_b = move_of(b);
                }
                End::Forbidden(p) => {
                    result.end_kind = END_FORBIDDEN;
                    result.end_a = move_of(p);
                }
                End::Unknown => {
                    result.end_kind = END_UNKNOWN;
                }
            }
        }
        SolveResult::Disproven => {
            result.status = STATUS_DISPROVEN;
            result.message("no VCF within max_attack_plies");
        }
        SolveResult::Aborted => {
            result.status = STATUS_BUDGET_EXHAUSTED;
            result.message("node budget exhausted before the search finished");
        }
    }
    result
}

/// Runs one query under a wall-clock limit by abandoning the search thread.
/// Upstream has no clock (it must compile for wasm), so this is the only way
/// to bound it in time. A timed-out search never reports a proof and keeps
/// running in the background until its node budget runs out.
fn run_with_deadline(query: VcfBridgeQuery) -> VcfBridgeResult {
    let deadline = Duration::from_millis(u64::from(query.time_limit_ms));
    match with_deadline(deadline, move || run(&query)) {
        Some(result) => result,
        None => {
            let mut result = VcfBridgeResult::empty();
            result.status = STATUS_INTERRUPTED;
            result.elapsed_ms = query.time_limit_ms;
            result.message(
                "wall-clock limit reached; the upstream search was abandoned, result unknown",
            );
            result
        }
    }
}

/// Runs `task` in a worker thread and waits at most `deadline`. Returns
/// `None` when the deadline passed first; the thread is then abandoned.
fn with_deadline<F>(deadline: Duration, task: F) -> Option<VcfBridgeResult>
where
    F: FnOnce() -> VcfBridgeResult + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(task());
    });
    receiver.recv_timeout(deadline).ok()
}

/// Answers one VCF question. Returns the status code and fills `out`; `out`
/// is null-safe but then only the status code is returned.
///
/// # Safety
/// `query` and `out` must be null or point to objects of the declared types.
#[no_mangle]
pub unsafe extern "C" fn vcf_bridge_solve(
    query: *const VcfBridgeQuery,
    out: *mut VcfBridgeResult,
) -> i32 {
    if out.is_null() {
        return STATUS_INVALID_INPUT;
    }
    unsafe { out.write(VcfBridgeResult::empty()) };
    if query.is_null() {
        let mut result = VcfBridgeResult::empty();
        result.status = STATUS_INVALID_INPUT;
        result.message("null query");
        unsafe { out.write(result) };
        return STATUS_INVALID_INPUT;
    }
    let owned = unsafe { query.read() };
    let result = if owned.time_limit_ms == 0 {
        run(&owned)
    } else {
        run_with_deadline(owned)
    };
    let status = result.status;
    unsafe { out.write(result) };
    status
}

/// Upstream identity: name, pinned commit and license.
#[no_mangle]
pub extern "C" fn vcf_bridge_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

/// Longest variation, in points, that the bridge can report.
#[no_mangle]
pub extern "C" fn vcf_bridge_max_path() -> i32 {
    MAX_PATH as i32
}


#[cfg(test)]
mod tests {
    use super::*;

    /// "H8,I8" -> upstream points. Row 8 is y = 7, as in the engine.
    fn parse_points(text: &str) -> Vec<Point> {
        text.split(',')
            .filter(|token| token.len() >= 2)
            .map(|token| {
                let x = token.as_bytes()[0] - b'A';
                let y: u8 = token[1..].parse::<u8>().unwrap() - 1;
                Point(x, y)
            })
            .collect()
    }

    fn query(blacks: &str, whites: &str, side_to_move: u8) -> VcfBridgeQuery {
        let mut query = VcfBridgeQuery {
            cells: [0; 225],
            side_to_move,
            rule: 2,
            max_attack_plies: 10,
            _padding: 0,
            max_nodes: 50_000,
            time_limit_ms: 0,
        };
        for point in parse_points(blacks) {
            query.cells[usize::from(point.1) * 15 + usize::from(point.0)] = 1;
        }
        for point in parse_points(whites) {
            query.cells[usize::from(point.1) * 15 + usize::from(point.0)] = 2;
        }
        query
    }

    const COUNTER_BLACK: &str = "H8,J8,K8,I7,L7,I6";
    const COUNTER_WHITE: &str = "H9,L8,J7,K6,I5,L5";

    #[test]
    fn counter_four_line_is_proven_with_engine_indices() {
        let result = run(&query(COUNTER_BLACK, COUNTER_WHITE, 1));
        assert_eq!(result.status, STATUS_PROVEN);
        assert_eq!(result.first_move, 7 * 15 + 8); // I8
        assert_eq!(result.path_len, 5);
        // I8, G8, I10, I9, J9 in the engine's y * 15 + x encoding.
        assert_eq!(result.path[..5], [113, 111, 143, 128, 129]);
        assert_eq!(result.end_kind, END_FOURS);
        assert!(result.nodes > 0);
    }

    #[test]
    fn budget_exhaustion_is_not_a_disproof() {
        let mut request = query(COUNTER_BLACK, COUNTER_WHITE, 1);
        request.max_nodes = 1;
        let result = run(&request);
        assert_eq!(result.status, STATUS_BUDGET_EXHAUSTED);
        assert_eq!(result.path_len, 0);
        assert_ne!(result.status, STATUS_DISPROVEN);
    }

    #[test]
    fn deadline_reports_interruption() {
        let interrupted = with_deadline(Duration::from_millis(1), || {
            thread::sleep(Duration::from_millis(50));
            VcfBridgeResult::empty()
        });
        assert!(interrupted.is_none());
    }

    #[test]
    fn invalid_cells_and_rules_are_reported() {
        let mut request = query(COUNTER_BLACK, COUNTER_WHITE, 1);
        request.cells[0] = 9;
        assert_eq!(run(&request).status, STATUS_INVALID_INPUT);

        let mut request = query(COUNTER_BLACK, COUNTER_WHITE, 1);
        request.side_to_move = 0;
        assert_eq!(run(&request).status, STATUS_INVALID_INPUT);

        let mut request = query(COUNTER_BLACK, COUNTER_WHITE, 1);
        request.rule = 1; // freestyle: upstream models Renju only
        assert_eq!(run(&request).status, STATUS_UNSUPPORTED);
    }

    #[test]
    fn null_pointers_are_invalid_input() {
        let mut out = VcfBridgeResult::empty();
        assert_eq!(
            unsafe { vcf_bridge_solve(std::ptr::null(), &mut out) },
            STATUS_INVALID_INPUT
        );
        let request = query(COUNTER_BLACK, COUNTER_WHITE, 1);
        assert_eq!(
            unsafe { vcf_bridge_solve(&request, std::ptr::null_mut()) },
            STATUS_INVALID_INPUT
        );
    }

    #[test]
    fn version_identifies_the_pinned_upstream() {
        let version = format!("{}", std::str::from_utf8(VERSION).unwrap().trim_end_matches('\0'));
        assert!(version.starts_with("quintet@"));
    }
}
