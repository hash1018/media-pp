//! Activating a capture on one process's audio instead of a device's.
//!
//! Windows calls this process loopback: a virtual device that mixes what one
//! process tree plays and nothing else, so a capture of a game does not also
//! carry the chat program next to it. It is activated rather than opened —
//! `IMMDeviceEnumerator` never sees it — and the activation is asynchronous,
//! which is the whole reason this module exists.
//!
//! Windows 10 2004 is where this appeared. An older build answers the
//! activation with an error, which is what a caller reports rather than
//! something this checks a version for.

use std::{
    mem, ptr,
    sync::mpsc::{self, Sender},
    time::Duration,
};

use windows::{
    Win32::{
        Foundation::{E_OUTOFMEMORY, E_POINTER, RPC_E_TIMEOUT},
        Media::Audio::{
            AUDIOCLIENT_ACTIVATION_PARAMS, AUDIOCLIENT_ACTIVATION_PARAMS_0,
            AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
            ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
            IActivateAudioInterfaceCompletionHandler,
            IActivateAudioInterfaceCompletionHandler_Impl, IAudioClient,
            PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        },
        System::{
            Com::{BLOB, CoTaskMemAlloc, StructuredStorage::PROPVARIANT},
            Variant::VT_BLOB,
        },
    },
    core::{Interface, Ref, implement},
};

/// How long [`activate`] waits for Windows to answer.
///
/// Bounded rather than blocking for ever: the completion handler runs on a
/// thread of the system's own, and a caller opening a Source deserves an
/// error it can show rather than a pipeline that never finishes being built.
/// Generous, because this is a one-off at open time — a machine slow enough
/// to need five seconds for it has already failed.
const ACTIVATION_TIMEOUT: Duration = Duration::from_secs(5);

/// What Windows calls when the activation is done, which is the only way to
/// learn that it is: it takes an object, not a handle to wait on.
#[implement(IActivateAudioInterfaceCompletionHandler)]
struct Completion(Sender<()>);

impl IActivateAudioInterfaceCompletionHandler_Impl for Completion_Impl {
    /// Nothing but a wake-up. What the activation actually produced is read
    /// from the operation by [`activate`], on its own thread, so nothing
    /// here has to be carried across one.
    fn ActivateCompleted(
        &self,
        _operation: Ref<IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        let _ = self.0.send(());
        Ok(())
    }
}

/// An `IAudioClient` that captures what `process_id` and the processes it
/// started are playing.
///
/// The tree, not the one process: a game that plays its sound through a
/// child, and a browser that gives every tab its own process, would
/// otherwise capture as silence. The caller still has to `Initialize` the
/// client with a format of its own — process loopback has no mix format to
/// ask for, since there is no endpoint behind it.
pub(crate) fn activate(process_id: u32) -> windows::core::Result<IAudioClient> {
    // The parameters are handed over, not lent: measured on Windows 11, the
    // activation frees this blob itself once it has read it. So it is
    // allocated with `CoTaskMemAlloc` — the allocator whose `CoTaskMemFree`
    // will be called on it — and never freed here. Keeping it on the stack,
    // as Microsoft's own sample does, corrupts the heap the moment Windows
    // lets go of it.
    // SAFETY: the size is this type's own, and the allocation is written
    // exactly once below before anything reads it.
    let params: *mut AUDIOCLIENT_ACTIVATION_PARAMS =
        unsafe { CoTaskMemAlloc(mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>()) }.cast();
    if params.is_null() {
        return Err(windows::core::Error::new(
            E_OUTOFMEMORY,
            "no room for the process loopback parameters",
        ));
    }
    // SAFETY: `params` is a live allocation of exactly this size, and this is
    // the only write to it.
    unsafe {
        ptr::write(
            params,
            AUDIOCLIENT_ACTIVATION_PARAMS {
                ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
                Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                    ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                        TargetProcessId: process_id,
                        ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                    },
                },
            },
        );
    }

    // A `PROPVARIANT` carrying those parameters as a blob, which is how this
    // one API takes them. The variant itself is only read — it is the blob
    // inside it that is taken — so it stays on the stack.
    let mut variant: PROPVARIANT = PROPVARIANT::default();
    // SAFETY: the variant was zeroed by `default`, so writing its tag and
    // the matching union member is the only way it is ever read.
    unsafe {
        let inner = &mut *variant.Anonymous.Anonymous;
        inner.vt = VT_BLOB;
        inner.Anonymous.blob = BLOB {
            cbSize: mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
            pBlobData: params.cast(),
        };
    }

    let (sender, done) = mpsc::channel();
    let handler: IActivateAudioInterfaceCompletionHandler = Completion(sender).into();
    // SAFETY: the device path is the constant Windows documents for process
    // loopback, the riid matches the interface the result is cast to, and
    // both the variant and the handler stay alive until the wait below
    // finishes.
    let operation = unsafe {
        ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some(&raw const variant),
            &handler,
        )?
    };

    if done.recv_timeout(ACTIVATION_TIMEOUT).is_err() {
        return Err(windows::core::Error::new(
            RPC_E_TIMEOUT,
            "the process audio capture was never activated",
        ));
    }

    let mut activation = windows::core::HRESULT(0);
    let mut client = None;
    // SAFETY: the operation is live and has completed — the handler above is
    // what said so — and both out-parameters are owned by this frame.
    unsafe { operation.GetActivateResult(&raw mut activation, &raw mut client) }?;
    activation.ok()?;
    client
        .ok_or_else(|| {
            windows::core::Error::new(E_POINTER, "the activation produced no audio client")
        })?
        .cast()
}
