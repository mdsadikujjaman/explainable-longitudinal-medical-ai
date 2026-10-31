"""One versioned identity recipe for BOTH workstreams; no IDs in diagnostics.

Includes cross-platform pseudonymization secret-file permission validation:
- POSIX: Strict mode 0600 (owner read/write, zero group/other access).
- Windows: Native NTFS security descriptor validation using explicit allowlist policy,
  handle-based inspection, symlink/reparse-point rejection, and parent-directory safeguards.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import sys


def subject_key(site: str, participant: str, secret: bytes) -> str:
    """Generate pseudonymized subject key using HMAC-SHA256."""
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("A new private secret of at least 32 bytes is required")
    blob_header = ("blob " + str(len(secret)) + "\0").encode()
    if hashlib.sha1(blob_header + secret).hexdigest() == "13ce176f2ec3bef436b85f2d7f3ab2b0d53dc292":
        raise ValueError("The publicly tracked pseudonymization secret must be replaced")
    if any(not isinstance(v, str) or not v or v != v.strip() for v in (site, participant)):
        raise ValueError("Site and participant must be nonempty canonical strings")
    payload = json.dumps(
        ["nacc74-site-participant-v1", site, participant],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf8")
    return "n74_" + hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _get_data_blocked_error(reason: str) -> Exception:
    """Retrieve or construct a DataBlocked exception without circular imports."""
    try:
        from nacc_progression.integration import DataBlocked

        return DataBlocked(reason)
    except Exception:
        return ValueError(reason)


# ============================================================================
# Windows NTFS Security Validation Helpers (Standard Library ctypes)
# ============================================================================

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _advapi32 = ctypes.windll.advapi32
    _kernel32 = ctypes.windll.kernel32

    # Windows Security and File Constants
    _READ_CONTROL = 0x00020000
    _FILE_READ_ATTRIBUTES = 0x0080
    _OPEN_EXISTING = 3
    _FILE_SHARE_READ = 1
    _FILE_SHARE_WRITE = 2
    _FILE_SHARE_DELETE = 4
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x0400

    _SE_FILE_OBJECT = 1
    _OWNER_SECURITY_INFORMATION = 1
    _DACL_SECURITY_INFORMATION = 4

    _ACCESS_ALLOWED_ACE_TYPE = 0x00
    _ACCESS_DENIED_ACE_TYPE = 0x01

    # Directory modification rights: writing, creating subfiles, deleting children, modifying permissions
    _DIR_MODIFY_MASK = (
        0x00000002  # FILE_ADD_FILE
        | 0x00000004  # FILE_ADD_SUBDIRECTORY
        | 0x00000040  # FILE_DELETE_CHILD
        | 0x00010000  # DELETE
        | 0x00040000  # WRITE_DAC
        | 0x00080000  # WRITE_OWNER
        | 0x40000000  # GENERIC_WRITE
        | 0x10000000  # GENERIC_ALL
    )

    # Configure ctypes function signatures
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    _advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    _advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    _kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]

    class _ACL_SIZE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("AceCount", wintypes.DWORD),
            ("AclBytesInUse", wintypes.DWORD),
            ("AclBytesFree", wintypes.DWORD),
        ]

    def _get_current_process_user_sid() -> str:
        """Query the current running process token to determine the user's SID."""
        token = wintypes.HANDLE()
        if not _advapi32.OpenProcessToken(_kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise RuntimeError("Failed to open current process token for security verification")
        try:
            needed = wintypes.DWORD()
            _advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
            buf = ctypes.create_string_buffer(needed.value)
            if not _advapi32.GetTokenInformation(token, 1, buf, needed.value, ctypes.byref(needed)):
                raise RuntimeError("Failed to query user token information")
            p_sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p)).contents
            str_sid = wintypes.LPWSTR()
            if not _advapi32.ConvertSidToStringSidW(p_sid, ctypes.byref(str_sid)):
                raise RuntimeError("Failed to convert process user SID to string format")
            sid_value = str_sid.value
            _kernel32.LocalFree(str_sid)
            return sid_value
        finally:
            _kernel32.CloseHandle(token)

    def _inspect_handle_security(
        handle: wintypes.HANDLE,
        is_directory: bool,
        trusted_sids: set[str],
        trusted_owners: set[str],
    ) -> None:
        """Inspect NTFS Security Descriptor on an open handle using an explicit allowlist policy."""
        p_owner = ctypes.c_void_p()
        p_dacl = ctypes.c_void_p()
        p_sd = ctypes.c_void_p()

        status = _advapi32.GetSecurityInfo(
            handle,
            _SE_FILE_OBJECT,
            _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION,
            ctypes.byref(p_owner),
            None,
            ctypes.byref(p_dacl),
            None,
            ctypes.byref(p_sd),
        )
        if status != 0:
            raise _get_data_blocked_error(
                f"Failed to query security descriptor on target handle (WinError code {status})"
            )

        try:
            # 1. Validate Owner Identity Policy
            if not p_owner.value:
                raise _get_data_blocked_error("Security descriptor has no defined owner")
            str_owner = wintypes.LPWSTR()
            if not _advapi32.ConvertSidToStringSidW(p_owner, ctypes.byref(str_owner)):
                raise _get_data_blocked_error("Failed to convert owner SID string")
            owner_sid = str_owner.value
            _kernel32.LocalFree(str_owner)

            if owner_sid not in trusted_owners:
                raise _get_data_blocked_error(
                    f"File owner ({owner_sid}) is not permitted by trusted identity policy"
                )

            # 2. Reject NULL or absent DACLs (world-writable in Windows security model)
            if not p_dacl.value:
                raise _get_data_blocked_error(
                    "Security descriptor has NULL or absent DACL (unrestricted access)"
                )

            size_info = _ACL_SIZE_INFORMATION()
            if not _advapi32.GetAclInformation(
                p_dacl, ctypes.byref(size_info), ctypes.sizeof(size_info), 2
            ):
                raise _get_data_blocked_error("Failed to query ACL size information")

            # 3. Inspect every Access Control Entry (ACE)
            for i in range(size_info.AceCount):
                p_ace = ctypes.c_void_p()
                if not _advapi32.GetAce(p_dacl, i, ctypes.byref(p_ace)):
                    raise _get_data_blocked_error(f"Failed to retrieve ACE index {i}")

                ace_header_ptr = ctypes.cast(p_ace, ctypes.POINTER(ctypes.c_ubyte))
                ace_type = ace_header_ptr[0]

                # Reject unsupported ACE types that cannot be safely evaluated
                if ace_type not in (_ACCESS_ALLOWED_ACE_TYPE, _ACCESS_DENIED_ACE_TYPE):
                    raise _get_data_blocked_error(
                        f"Unsupported ACE type ({ace_type}) encountered in security descriptor"
                    )

                mask_ptr = ctypes.cast(p_ace.value + 4, ctypes.POINTER(wintypes.DWORD))
                mask = mask_ptr.contents.value

                sid_ptr = p_ace.value + 8
                str_ace_sid = wintypes.LPWSTR()
                if not _advapi32.ConvertSidToStringSidW(ctypes.c_void_p(sid_ptr), ctypes.byref(str_ace_sid)):
                    raise _get_data_blocked_error(f"Failed to convert ACE SID {i} to string")
                ace_sid = str_ace_sid.value
                _kernel32.LocalFree(str_ace_sid)

                # Explicit Allowlist Enforcement
                if ace_type == _ACCESS_ALLOWED_ACE_TYPE:
                    if ace_sid not in trusted_sids:
                        if not is_directory:
                            # The secret file itself must not grant ANY permissions to untrusted principals
                            if mask != 0:
                                raise _get_data_blocked_error(
                                    f"Secret file grants access to unauthorized principal ({ace_sid})"
                                )
                        else:
                            # The parent directory must not grant modification/replacement permissions
                            if mask & _DIR_MODIFY_MASK:
                                raise _get_data_blocked_error(
                                    f"Secret parent directory permits modification by unauthorized principal ({ace_sid})"
                                )
        finally:
            if p_sd.value:
                _kernel32.LocalFree(p_sd)

    def _validate_windows_secret_file_security(secret_path: Path) -> None:
        """Validate secret file and parent directory on Windows NTFS."""
        resolved = secret_path.resolve()

        # Handle symlink check before handle opening
        if secret_path.is_symlink():
            raise _get_data_blocked_error("Secret file must not be a symbolic link")

        current_user_sid = _get_current_process_user_sid()
        trusted_owners = {
            current_user_sid,
            "S-1-5-32-544",  # BUILTIN\Administrators
            "S-1-5-18",  # NT AUTHORITY\SYSTEM
        }
        trusted_principals = {
            current_user_sid,
            "S-1-5-32-544",  # BUILTIN\Administrators
            "S-1-5-18",  # NT AUTHORITY\SYSTEM
            "S-1-3-4",  # OWNER RIGHTS
            "S-1-3-0",  # CREATOR OWNER (container inheritance placeholder)
        }

        # 1. Open secret file handle with open-reparse-point flag to prevent TOCTOU substitution
        h_file = _kernel32.CreateFileW(
            str(resolved),
            _READ_CONTROL | _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if h_file == wintypes.HANDLE(-1).value or h_file == 0:
            raise _get_data_blocked_error(
                f"Could not open secret file handle for verification (WinError {ctypes.GetLastError()})"
            )

        try:
            info = _BY_HANDLE_FILE_INFORMATION()
            if not _kernel32.GetFileInformationByHandle(h_file, ctypes.byref(info)):
                raise _get_data_blocked_error("Failed to query secret file handle information")
            if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise _get_data_blocked_error("Secret file must not be a reparse point or junction")

            _inspect_handle_security(
                h_file,
                is_directory=False,
                trusted_sids=trusted_principals,
                trusted_owners=trusted_owners,
            )
        finally:
            _kernel32.CloseHandle(h_file)

        # 2. Inspect Parent Directory to prevent replacement/tampering
        parent = resolved.parent
        h_dir = _kernel32.CreateFileW(
            str(parent),
            _READ_CONTROL | _FILE_READ_ATTRIBUTES,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        if h_dir == wintypes.HANDLE(-1).value or h_dir == 0:
            raise _get_data_blocked_error(
                f"Could not open secret parent directory handle (WinError {ctypes.GetLastError()})"
            )

        try:
            dir_info = _BY_HANDLE_FILE_INFORMATION()
            if not _kernel32.GetFileInformationByHandle(h_dir, ctypes.byref(dir_info)):
                raise _get_data_blocked_error("Failed to query secret parent directory handle information")
            if dir_info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise _get_data_blocked_error(
                    "Secret parent directory must not be a reparse point or junction"
                )

            _inspect_handle_security(
                h_dir,
                is_directory=True,
                trusted_sids=trusted_principals,
                trusted_owners=trusted_owners,
            )
        finally:
            _kernel32.CloseHandle(h_dir)


# ============================================================================
# Public Cross-Platform Entrypoint
# ============================================================================


def _get_posix_file_mode(file_path: Path) -> int:
    """Retrieve POSIX st_mode for permission validation.

    Isolated helper to allow clean platform testing without monkeypatching
    the global pathlib.Path.stat method.
    """
    return file_path.stat().st_mode


def validate_secret_file_security(secret_path: Path | str, repo_root: Path | str | None = None) -> None:
    """Validate pseudonymization secret permissions and location across platforms.

    Rules:
    - Path must be absolute.
    - Path must not reside within the repository directory.
    - File must exist and be a regular file.
    - On POSIX (Linux/macOS): Strict 0600 mode (no group or other access bits: st_mode & 0o077 == 0).
    - On Windows (NTFS): Validates Security Descriptor using explicit allowlist policy,
      checks file and parent directory DACLs, and blocks symlinks and reparse points.
    """
    raw_path = Path(secret_path)
    if not raw_path.is_absolute():
        raise _get_data_blocked_error("Secret must be an absolute private path with mode 0600")

    resolved = raw_path.resolve()
    if repo_root is not None:
        repo = Path(repo_root).resolve()
        if resolved == repo or repo in resolved.parents:
            raise _get_data_blocked_error("Secret must be outside the repository and have mode 0600")

    if not resolved.exists() or not resolved.is_file():
        raise _get_data_blocked_error("Secret file does not exist or is not a regular file")

    if sys.platform != "win32":
        # Strict POSIX mode 0600 validation
        mode = _get_posix_file_mode(resolved)
        if mode & 0o077:
            raise _get_data_blocked_error("Secret must be an absolute private path with mode 0600")
    else:
        # Native Windows NTFS Security Descriptor validation
        _validate_windows_secret_file_security(resolved)
