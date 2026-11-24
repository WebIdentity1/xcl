//! Write32/64BitValueBottomOfPipeX into memory the GPU does not see: done by the CPU once the GPU
//! has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Here each queue gets a fence and completion worker of XCL's: after a submission whose lists
//! carry such writes the queue signals its fence, and its worker thread does the writes in
//! submission order when the GPU is there – what bottom of pipe means, without one stalled queue
//! blocking another queue's writes.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Threading as th;

/// A deferred CPU write into title memory (plain values, OS QPC timestamps, or GPU readback
/// slots copied atomically once the queue's fence completes).
#[derive(Clone)]
pub(super) enum Write {
    U32(usize, u32),
    U64(usize, u64),
    OsTimestamp(usize),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    writes: Vec<Write>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence, the last value signalled, and its dedicated completion worker
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
    tx: mpsc::Sender<Job>,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);
static NEXT_WORKER_ID: AtomicU32 = AtomicU32::new(0);

/// how long a queue's worker waits for one submission before writing anyway (a removed device, a
/// queue that never gets there)
const WAIT_MS: u32 = 10_000;

fn spawn_queue_worker() -> mpsc::Sender<Job> {
    let (tx, rx) = mpsc::channel::<Job>();
    let id = NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new().name(format!("xcl-cpu-writes-{id}")).spawn(move || unsafe {
        let event = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
        for job in rx {
            if job.fence.GetCompletedValue() < job.value && job.fence.SetEventOnCompletion(job.value, HANDLE(event as *mut c_void)).is_ok() {
                th::WaitForSingleObject(event, WAIT_MS);
            }
            for w in &job.writes {
                write_now(w);
            }
        }
        if !event.is_null() {
            windows_sys::Win32::Foundation::CloseHandle(event);
        }
    });
    if let Err(e) = spawned {
        crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
    }
    tx
}

/// Drop the fence and worker channel for a released command queue so its worker exits once
/// any queued writes drain and a reused pointer starts fresh.
pub(super) fn forget_queue(queue: usize) {
    if let Some(m) = FENCES.lock().unwrap().as_mut() {
        m.remove(&queue);
    }
}

fn protect_writable(protect: u32) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    protect & PAGE_GUARD == 0 && protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
}

/// Perform `w` now if its destination is still committed, writable (not `PAGE_GUARD`),
/// and properly aligned for an atomic store.
pub(super) unsafe fn write_now(w: &Write) {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let (va, bytes) = match *w {
        Write::U32(va, _) => (va, 4usize),
        Write::U64(va, _) | Write::OsTimestamp(va) | Write::ReadbackU64 { va, .. } => (va, 8usize),
    };
    if va == 0 || !va.is_multiple_of(bytes) {
        return;
    }
    let Some(end) = va.checked_add(bytes) else { return };
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return;
    }
    if !protect_writable(m.Protect) || (m.BaseAddress as usize).saturating_add(m.RegionSize) < end {
        return;
    }
    match *w {
        Write::U32(va, v) => (*(va as *const AtomicU32)).store(v, Ordering::Release),
        Write::U64(va, v) => (*(va as *const AtomicU64)).store(v, Ordering::Release),
        Write::OsTimestamp(va) => (*(va as *const AtomicU64)).store(super::frame::qpc() as u64, Ordering::Release),
        Write::ReadbackU64 { va, src, .. } => {
            let value = (*(src as *const AtomicU64)).load(Ordering::Acquire);
            (*(va as *const AtomicU64)).store(value, Ordering::Release);
        }
    }
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    let Some(end) = va.checked_add(bytes) else { return false };
    crate::hle::kernelx::lazy::ensure(va, bytes);
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0;
    found && m.State == MEM_COMMIT && protect_writable(m.Protect) && (m.BaseAddress as usize).saturating_add(m.RegionSize) >= end
}

/// `writes` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    if writes.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let (tx, job) = {
        let mut g = FENCES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
                v.insert(QueueFence { fence, value: 0, tx: spawn_queue_worker() })
            }
        };
        entry.value += 1;
        if q.Signal(&entry.fence, entry.value).is_err() {
            return;
        }
        (entry.tx.clone(), Job { fence: entry.fence.clone(), value: entry.value, writes })
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len()));
    let _ = tx.send(job);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new(AtomicU64::new(0));
            let va = &*cell as *const AtomicU64 as usize;
            assert!(writable(va, 8));
            let want = 7u64 | (9u64 << 32);
            after(q.as_raw(), vec![Write::U64(va, want)]);
            let t = std::time::Instant::now();
            while cell.load(Ordering::Acquire) != want {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(cell.load(Ordering::Acquire), want);
            forget_queue(q.as_raw() as usize);
        }
    }

    /// OS QPC timestamp writes land a QPC value in range after the queue completes, and
    /// `PAGE_GUARD` pages are rejected without faulting.
    #[test]
    fn os_timestamp_writes_qpc_and_writable_rejects_page_guard() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_READWRITE};
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();

            let cell = Box::new(AtomicU64::new(0));
            let va = &*cell as *const AtomicU64 as usize;
            let before = super::super::frame::qpc() as u64;
            after(q.as_raw(), vec![Write::OsTimestamp(va)]);
            let t = std::time::Instant::now();
            while cell.load(Ordering::Acquire) == 0 {
                assert!(t.elapsed().as_secs() < 5, "OS timestamp never written");
                std::thread::yield_now();
            }
            let after_qpc = super::super::frame::qpc() as u64;
            let got = cell.load(Ordering::Acquire);
            assert!(got >= before && got <= after_qpc, "QPC {got} outside [{before}, {after_qpc}]");

            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            assert!(!page.is_null());
            let mut old = 0u32;
            assert_ne!(VirtualProtect(page, 4096, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            assert!(!writable(page as usize, 8), "PAGE_GUARD must not be treated as writable");
            write_now(&Write::U64(page as usize, 0x1234));
            assert_ne!(VirtualProtect(page, 4096, PAGE_READWRITE, &mut old), 0);
            assert_eq!((*(page as *const AtomicU64)).load(Ordering::Acquire), 0);
            VirtualFree(page, 0, MEM_RELEASE);
            forget_queue(q.as_raw() as usize);
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
            forget_queue(q.as_raw() as usize);
        }
    }

    /// A queue waiting on an unsignaled fence must not head-of-line block another queue's
    /// bottom-of-pipe writes.
    #[test]
    fn blocked_queue_does_not_stall_another_queues_writes() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q1: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let q2: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let blocker: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            q1.Wait(&blocker, 1).unwrap();

            let cell1 = Box::new(AtomicU32::new(0));
            let cell2 = Box::new(AtomicU32::new(0));
            let va1 = cell1.as_ref() as *const AtomicU32 as usize;
            let va2 = cell2.as_ref() as *const AtomicU32 as usize;
            after(q1.as_raw(), vec![Write::U32(va1, 11)]);
            after(q2.as_raw(), vec![Write::U32(va2, 22)]);

            let t = std::time::Instant::now();
            while cell2.load(Ordering::Acquire) != 22 && t.elapsed().as_secs() < 2 {
                std::thread::yield_now();
            }
            let independent = cell2.load(Ordering::Acquire) == 22;
            let before_signal = cell1.load(Ordering::Acquire);
            // Unblock and drain before assertions, including the old global-worker control.
            blocker.Signal(1).unwrap();
            let t = std::time::Instant::now();
            while (cell1.load(Ordering::Acquire) != 11 || cell2.load(Ordering::Acquire) != 22) && t.elapsed().as_secs() < 5 {
                std::thread::yield_now();
            }
            let drained = cell1.load(Ordering::Acquire) == 11 && cell2.load(Ordering::Acquire) == 22;
            forget_queue(q1.as_raw() as usize);
            forget_queue(q2.as_raw() as usize);
            if !drained {
                // Keep the destinations alive if a completion worker is still using them.
                std::mem::forget(cell1);
                std::mem::forget(cell2);
            }
            println!("queue independence={independent}, q1 before signal={before_signal}, drained={drained}, device={:?}", dev.GetDeviceRemovedReason());
            assert!(drained, "writes never landed after unblocking");
            assert!(independent, "q2 write stalled behind blocked q1");
            assert_eq!(before_signal, 0, "q1 write landed before its fence was signaled");
        }
    }
}
