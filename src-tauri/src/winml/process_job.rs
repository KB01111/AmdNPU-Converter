use std::{
    io,
    mem::size_of,
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    ptr,
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    Foundation::INVALID_HANDLE_VALUE,
    System::{
        Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
        },
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectBasicAccountingInformation,
            JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject,
            TerminateJobObject, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
        Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME},
    },
};

/// The handle is not inheritable: dropping the worker future or exiting the app kills its whole tree.
pub(super) struct ProcessJob(OwnedHandle);

impl ProcessJob {
    pub fn new() -> io::Result<Self> {
        // SAFETY: null security attributes create an unnamed, non-inheritable job handle.
        let handle = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: CreateJobObjectW returned a new handle owned exclusively by this value.
        let job = Self(unsafe { OwnedHandle::from_raw_handle(handle) });
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: handle and correctly sized limit structure remain valid for the call.
        if unsafe {
            SetInformationJobObject(
                job.0.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(job)
    }

    pub fn assign_and_resume(&self, child: &tokio::process::Child) -> io::Result<()> {
        let process = child
            .raw_handle()
            .ok_or_else(|| io::Error::other("Worker process handle is unavailable"))?;
        let pid = child
            .id()
            .ok_or_else(|| io::Error::other("Worker process ID is unavailable"))?;
        // SAFETY: the child was created suspended and both borrowed handles are valid.
        if unsafe { AssignProcessToJobObject(self.0.as_raw_handle(), process) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // CREATE_SUSPENDED prevents the primary thread from launching descendants before assignment.
        // SAFETY: Toolhelp returns an owned snapshot; TH32CS_SNAPTHREAD ignores the PID argument.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: snapshot is a newly created valid handle.
        let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot) };
        let mut entry = THREADENTRY32 {
            dwSize: size_of::<THREADENTRY32>() as u32,
            ..Default::default()
        };
        // SAFETY: the snapshot and writable THREADENTRY32 are valid throughout enumeration.
        let mut present = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) };
        while present != 0 {
            if entry.th32OwnerProcessID == pid {
                // SAFETY: request only resume access and keep this handle non-inheritable.
                let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
                if thread.is_null() {
                    return Err(io::Error::last_os_error());
                }
                // SAFETY: OpenThread returned a new valid owned handle.
                let thread = unsafe { OwnedHandle::from_raw_handle(thread) };
                // SAFETY: this is the suspended worker's primary thread, assigned to the job above.
                match unsafe { ResumeThread(thread.as_raw_handle()) } {
                    1 => return Ok(()),
                    u32::MAX => return Err(io::Error::last_os_error()),
                    _ => {
                        return Err(io::Error::other(
                            "Worker primary thread was not suspended exactly once",
                        ))
                    }
                }
            }
            entry.dwSize = size_of::<THREADENTRY32>() as u32;
            // SAFETY: same valid snapshot and correctly sized output buffer.
            present = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) };
        }
        Err(io::Error::other(
            "Suspended worker primary thread was not found",
        ))
    }

    pub async fn terminate_and_wait(&self) -> io::Result<()> {
        // SAFETY: this job owns only this worker and its descendants.
        if unsafe { TerminateJobObject(self.0.as_raw_handle(), 1) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let mut accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
            // SAFETY: the owned handle and correctly sized writable accounting buffer are valid.
            if unsafe {
                QueryInformationJobObject(
                    self.0.as_raw_handle(),
                    JobObjectBasicAccountingInformation,
                    (&mut accounting as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast(),
                    size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                    ptr::null_mut(),
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            if accounting.ActiveProcesses == 0 {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::other(
                    "Worker process tree did not stop within 10 seconds",
                ));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
