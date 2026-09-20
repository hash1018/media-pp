//! Which processes are playing sound, for a per-process capture's picker.
//!
//! An audio session is what a process gets the first time it opens a render
//! stream, and it outlives the sound: an application that played something a
//! minute ago still has one. So this is the list of applications that *use*
//! the sound card, not only those making a noise this instant — which is
//! what a picker wants, since somebody choosing which game to capture is
//! usually doing it between rounds rather than mid-explosion.

use std::collections::BTreeMap;

use windows::Win32::{
    Foundation::{CloseHandle, S_OK},
    Media::Audio::{
        AudioSessionStateExpired, DEVICE_STATE_ACTIVE, IAudioSessionControl2,
        IAudioSessionManager2, IMMDeviceEnumerator, MMDeviceEnumerator, eRender,
    },
    System::{
        Com::{CLSCTX_ALL, CoCreateInstance},
        Threading::{
            OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
            QueryFullProcessImageNameW,
        },
    },
};

use windows::core::Interface;

use super::super::com::ComApartment;

/// One process with an audio session of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasapiProcess {
    /// The process id, which is what a capture is opened against — and only
    /// for as long as that process lives. A caller that wants to find the
    /// same application again after it has been restarted should remember
    /// [`WasapiProcess::executable`] and look the id up again.
    pub id: u32,
    /// The executable's own file name, extension and all — `chrome.exe`.
    /// Empty where the process refused to say, which a caller can still
    /// capture by id.
    pub executable: String,
}

/// Every process with a render session on any active playback endpoint,
/// by process id.
///
/// Deduplicated: one process playing to two endpoints, or holding several
/// sessions on one, is one entry — a capture takes the process, not the
/// session. Expired sessions are left out, since the process they belonged
/// to has gone; Windows' own "System Sounds" session is left out too,
/// because it belongs to no process a caller could name.
pub(crate) fn list_processes() -> windows::core::Result<Vec<WasapiProcess>> {
    let _apartment = ComApartment::new()?;
    // SAFETY: COM is initialized on this thread and the registered class is
    // requested as its documented `IMMDeviceEnumerator` interface.
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)? };

    // Sorted by id so a picker's list does not reorder itself between two
    // looks at an unchanged machine.
    let mut processes: BTreeMap<u32, String> = BTreeMap::new();
    // SAFETY: the enumerator is live and both the flow and state flags are
    // documented values.
    let endpoints = unsafe { enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)? };
    // SAFETY: `endpoints` is live and `GetCount` takes no pointers.
    let count = unsafe { endpoints.GetCount()? };
    for index in 0..count {
        // SAFETY: `index` is bounded by the count of this same live
        // collection.
        let Ok(endpoint) = (unsafe { endpoints.Item(index) }) else {
            continue;
        };
        // SAFETY: `endpoint` is a live endpoint and the session manager is
        // requested as its documented interface. An endpoint that refuses
        // one — a device being removed as this runs — is skipped rather
        // than failing the whole list.
        let Ok(manager) = (unsafe { endpoint.Activate::<IAudioSessionManager2>(CLSCTX_ALL, None) })
        else {
            continue;
        };
        // SAFETY: `manager` is live; the enumerator it returns owns its own
        // COM reference.
        let Ok(sessions) = (unsafe { manager.GetSessionEnumerator() }) else {
            continue;
        };
        // SAFETY: `sessions` is live and `GetCount` takes no pointers.
        let Ok(session_count) = (unsafe { sessions.GetCount() }) else {
            continue;
        };
        for session in 0..session_count {
            // SAFETY: `session` is bounded by the count above, and the
            // control is queried for its own documented extension interface.
            let Ok(control) = (unsafe { sessions.GetSession(session) }) else {
                continue;
            };
            let Ok(control) = control.cast::<IAudioSessionControl2>() else {
                continue;
            };
            // SAFETY: `control` is live. `IsSystemSoundsSession` answers
            // `S_OK` for the session Windows plays its own notifications
            // through, which belongs to no process a caller could pick, and
            // `S_FALSE` for every other — two successes, which is why this
            // compares the value rather than asking whether it succeeded.
            if unsafe { control.IsSystemSoundsSession() } == S_OK {
                continue;
            }
            // SAFETY: as above; a state that cannot be read is treated as
            // one worth listing rather than one to drop.
            if unsafe { control.GetState() } == Ok(AudioSessionStateExpired) {
                continue;
            }
            // SAFETY: as above. A session with no process behind it is
            // reported as id zero, which is nothing to capture.
            let Ok(id) = (unsafe { control.GetProcessId() }) else {
                continue;
            };
            if id == 0 {
                continue;
            }
            processes
                .entry(id)
                .or_insert_with(|| executable_name(id).unwrap_or_default());
        }
    }

    Ok(processes
        .into_iter()
        .map(|(id, executable)| WasapiProcess { id, executable })
        .collect())
}

/// The file name of what `id` is running, or `None` where Windows will not
/// say — a process at a higher integrity level than this one, or one that
/// exited between being listed and being asked.
fn executable_name(id: u32) -> Option<String> {
    // SAFETY: the handle is closed on every path below, and
    // `QueryFullProcessImageNameW` writes at most `length` wide characters
    // into a buffer of that size and updates `length` to what it wrote.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, id).ok()?;
        let mut buffer = [0u16; 260];
        let mut length = buffer.len() as u32;
        let read = QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(buffer.as_mut_ptr()),
            &mut length,
        );
        let _ = CloseHandle(process);
        read.ok()?;
        let path = String::from_utf16_lossy(&buffer[..length as usize]);
        Some(
            path.rsplit(['\\', '/'])
                .next()
                .unwrap_or(path.as_str())
                .to_owned(),
        )
    }
}
