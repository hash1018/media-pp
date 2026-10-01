//! `IMFAttributes`, delegated: the activator, the source and the stream are
//! each an attribute store as well, and Media Foundation's own store does
//! the work.

/// Implements `IMFAttributes_Impl` for `$impl` by forwarding every method to
/// the `IMFAttributes` in its `attributes` field.
macro_rules! delegate_attributes {
    ($impl:ty) => {
        impl ::windows::Win32::Media::MediaFoundation::IMFAttributes_Impl for $impl {
            fn GetItem(
                &self,
                key: *const ::windows_core::GUID,
                value: *mut ::windows::Win32::System::Com::StructuredStorage::PROPVARIANT,
            ) -> ::windows_core::Result<()> {
                // SAFETY: forwarded exactly as the caller handed them.
                unsafe {
                    self.attributes
                        .GetItem(key, (!value.is_null()).then_some(value))
                }
            }
            fn GetItemType(
                &self,
                key: *const ::windows_core::GUID,
            ) -> ::windows_core::Result<::windows::Win32::Media::MediaFoundation::MF_ATTRIBUTE_TYPE>
            {
                // SAFETY: as above.
                unsafe { self.attributes.GetItemType(key) }
            }
            fn CompareItem(
                &self,
                key: *const ::windows_core::GUID,
                value: *const ::windows::Win32::System::Com::StructuredStorage::PROPVARIANT,
            ) -> ::windows_core::Result<::windows_core::BOOL> {
                // SAFETY: as above.
                unsafe { self.attributes.CompareItem(key, value) }
            }
            fn Compare(
                &self,
                theirs: ::windows_core::Ref<
                    ::windows::Win32::Media::MediaFoundation::IMFAttributes,
                >,
                match_type: ::windows::Win32::Media::MediaFoundation::MF_ATTRIBUTES_MATCH_TYPE,
            ) -> ::windows_core::Result<::windows_core::BOOL> {
                // SAFETY: as above.
                unsafe { self.attributes.Compare(theirs.as_ref(), match_type) }
            }
            fn GetUINT32(&self, key: *const ::windows_core::GUID) -> ::windows_core::Result<u32> {
                // SAFETY: as above.
                unsafe { self.attributes.GetUINT32(key) }
            }
            fn GetUINT64(&self, key: *const ::windows_core::GUID) -> ::windows_core::Result<u64> {
                // SAFETY: as above.
                unsafe { self.attributes.GetUINT64(key) }
            }
            fn GetDouble(&self, key: *const ::windows_core::GUID) -> ::windows_core::Result<f64> {
                // SAFETY: as above.
                unsafe { self.attributes.GetDouble(key) }
            }
            fn GetGUID(
                &self,
                key: *const ::windows_core::GUID,
            ) -> ::windows_core::Result<::windows_core::GUID> {
                // SAFETY: as above.
                unsafe { self.attributes.GetGUID(key) }
            }
            fn GetStringLength(
                &self,
                key: *const ::windows_core::GUID,
            ) -> ::windows_core::Result<u32> {
                // SAFETY: as above.
                unsafe { self.attributes.GetStringLength(key) }
            }
            fn GetString(
                &self,
                key: *const ::windows_core::GUID,
                value: ::windows_core::PWSTR,
                size: u32,
                length: *mut u32,
            ) -> ::windows_core::Result<()> {
                let store = &self.attributes;
                // SAFETY: the store's own vtable entry, with the caller's
                // buffer and its size passed through untouched.
                unsafe {
                    (::windows_core::Interface::vtable(store).GetString)(
                        ::windows_core::Interface::as_raw(store),
                        key,
                        value,
                        size,
                        length,
                    )
                    .ok()
                }
            }
            fn GetAllocatedString(
                &self,
                key: *const ::windows_core::GUID,
                value: *mut ::windows_core::PWSTR,
                length: *mut u32,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.GetAllocatedString(key, value, length) }
            }
            fn GetBlobSize(&self, key: *const ::windows_core::GUID) -> ::windows_core::Result<u32> {
                // SAFETY: as above.
                unsafe { self.attributes.GetBlobSize(key) }
            }
            fn GetBlob(
                &self,
                key: *const ::windows_core::GUID,
                buffer: *mut u8,
                size: u32,
                written: *mut u32,
            ) -> ::windows_core::Result<()> {
                let store = &self.attributes;
                // SAFETY: as for `GetString`.
                unsafe {
                    (::windows_core::Interface::vtable(store).GetBlob)(
                        ::windows_core::Interface::as_raw(store),
                        key,
                        buffer,
                        size,
                        written,
                    )
                    .ok()
                }
            }
            fn GetAllocatedBlob(
                &self,
                key: *const ::windows_core::GUID,
                buffer: *mut *mut u8,
                size: *mut u32,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.GetAllocatedBlob(key, buffer, size) }
            }
            fn GetUnknown(
                &self,
                key: *const ::windows_core::GUID,
                iid: *const ::windows_core::GUID,
                object: *mut *mut ::core::ffi::c_void,
            ) -> ::windows_core::Result<()> {
                let store = &self.attributes;
                // SAFETY: as for `GetString`.
                unsafe {
                    (::windows_core::Interface::vtable(store).GetUnknown)(
                        ::windows_core::Interface::as_raw(store),
                        key,
                        iid,
                        object,
                    )
                    .ok()
                }
            }
            fn SetItem(
                &self,
                key: *const ::windows_core::GUID,
                value: *const ::windows::Win32::System::Com::StructuredStorage::PROPVARIANT,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetItem(key, value) }
            }
            fn DeleteItem(&self, key: *const ::windows_core::GUID) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.DeleteItem(key) }
            }
            fn DeleteAllItems(&self) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.DeleteAllItems() }
            }
            fn SetUINT32(
                &self,
                key: *const ::windows_core::GUID,
                value: u32,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetUINT32(key, value) }
            }
            fn SetUINT64(
                &self,
                key: *const ::windows_core::GUID,
                value: u64,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetUINT64(key, value) }
            }
            fn SetDouble(
                &self,
                key: *const ::windows_core::GUID,
                value: f64,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetDouble(key, value) }
            }
            fn SetGUID(
                &self,
                key: *const ::windows_core::GUID,
                value: *const ::windows_core::GUID,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetGUID(key, value) }
            }
            fn SetString(
                &self,
                key: *const ::windows_core::GUID,
                value: &::windows_core::PCWSTR,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetString(key, *value) }
            }
            fn SetBlob(
                &self,
                key: *const ::windows_core::GUID,
                buffer: *const u8,
                size: u32,
            ) -> ::windows_core::Result<()> {
                let store = &self.attributes;
                // SAFETY: as for `GetString`.
                unsafe {
                    (::windows_core::Interface::vtable(store).SetBlob)(
                        ::windows_core::Interface::as_raw(store),
                        key,
                        buffer,
                        size,
                    )
                    .ok()
                }
            }
            fn SetUnknown(
                &self,
                key: *const ::windows_core::GUID,
                value: ::windows_core::Ref<::windows_core::IUnknown>,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.SetUnknown(key, value.as_ref()) }
            }
            fn LockStore(&self) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.LockStore() }
            }
            fn UnlockStore(&self) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.UnlockStore() }
            }
            fn GetCount(&self) -> ::windows_core::Result<u32> {
                // SAFETY: as above.
                unsafe { self.attributes.GetCount() }
            }
            fn GetItemByIndex(
                &self,
                index: u32,
                key: *mut ::windows_core::GUID,
                value: *mut ::windows::Win32::System::Com::StructuredStorage::PROPVARIANT,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe {
                    self.attributes
                        .GetItemByIndex(index, key, (!value.is_null()).then_some(value))
                }
            }
            fn CopyAllItems(
                &self,
                into: ::windows_core::Ref<::windows::Win32::Media::MediaFoundation::IMFAttributes>,
            ) -> ::windows_core::Result<()> {
                // SAFETY: as above.
                unsafe { self.attributes.CopyAllItems(into.as_ref()) }
            }
        }
    };
}

pub(crate) use delegate_attributes;

/// A new, empty attribute store.
pub(crate) fn store() -> windows_core::Result<windows::Win32::Media::MediaFoundation::IMFAttributes>
{
    let mut attributes = None;
    // SAFETY: `attributes` is a valid out-parameter for the call.
    unsafe { windows::Win32::Media::MediaFoundation::MFCreateAttributes(&mut attributes, 8)? };
    attributes.ok_or_else(|| windows_core::Error::from(windows::Win32::Foundation::E_POINTER))
}
