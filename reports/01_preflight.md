# Preflight Diagnostic Report: Prompt 01

**Date**: 2026-09-16  
**Host Architecture**: x86_64 (AMD64)  
**Host Model**: HP Pavilion Gaming Laptop PC 15-ec2xxx  
**Operating System**: Windows 11 Home (Build 26200.9457)  
**WSL Distro**: Ubuntu 26.04 LTS (Kernel: 6.18.33.2-microsoft-standard-WSL2)  
**Overall Verdict**: `HOST_PASS`

---

## 1. Execution Context & Terminal Verification
- **Host Confirmation**: Probes executed directly on physical Windows host via PowerShell (`pwsh`) and inside the native WSL2 instance (`Ubuntu`). Host hardware serial numbers and personal identifiers are omitted from this report to protect user privacy.
- **Virtualization Confirmation**: Execution is on the local laptop hardware, not an isolated cloud container or mock sandbox.
- **Unrelated Work Preservation**:
  - Existing `v1/` codebase (including tests, scripts, ingestion, and metric pipeline) remains completely untouched and preserved.
  - Existing project files (`README.md`, `CURRENT_APPROACH_DOCUMENTATION.md`, `.gitignore`, `data/`) inspected and preserved.
  - Git status check exited with code 1 (`fatal: not a git repository (or any of the parent directories): .git`). No existing git history or unrelated branches were disrupted.
- **WSL Project Workspace**:
  - Proposed WSL path: `~/projects/insta360_mapper`
  - Actual resolved path: `/home/rohit/projects/insta360_mapper`
  - Selected Distro: `Ubuntu` (Default, Version 2)
  - Filesystem Type: Native Linux `ext4` on `/dev/sdd` (high-performance POSIX I/O, independent of slow 9p/drvfs Windows mounts).

---

## 2. Windows PowerShell Diagnostics

Each diagnostic was executed individually with exit codes recorded:

| Diagnostic Command | Exit Code | Relevant Extracted Facts |
| :--- | :---: | :--- |
| `Get-CimInstance Win32_OperatingSystem` | `0` | Windows 11 Version 10.0.26200, Build 26200 |
| `Get-CimInstance Win32_ComputerSystem` | `0` | HP Pavilion Gaming Laptop PC 15-ec2xxx, Total RAM: 7,887,228,928 bytes (~7.35 GiB / 8.0 GB) |
| `Get-CimInstance Win32_Processor` | `0` | AMD Ryzen 5 5600H with Radeon Graphics (6 physical cores, 12 logical threads, base clock ~3.3 GHz) |
| `Get-Volume` | `0` | **C:** (Windows SSD): 94.11 GB free / 302.37 GB total (NTFS, Healthy)<br>**D:**: 32.87 GB free / 173.71 GB total (NTFS, Healthy) |
| `wsl.exe --version` | `0` | WSL version 2.7.13.0, Kernel 6.18.33.2-2, WSLg 1.0.73.2 |
| `wsl.exe --status` | `0` | Default Distribution: Ubuntu, Default Version: 2 |
| `wsl.exe --list --verbose` | `0` | `* Ubuntu` (Stopped/Running, Version 2), `docker-desktop` (Stopped, Version 2) |
| `nvidia-smi.exe` | `0` | GPU: NVIDIA GeForce GTX 1650 (Laptop), Driver: 581.83, CUDA Version Capability: 13.0, Total VRAM: 4096 MiB |
| `$env:USERPROFILE\.wslconfig` | `0` | Checked without modification; file does not exist (`FILE_NOT_FOUND`). Default WSL2 dynamic allocation is active. |

---

## 3. WSL2 (Ubuntu) Diagnostics

WSL commands executed inside target distribution `Ubuntu`:

| Diagnostic Command | Exit Code | Output Summary |
| :--- | :---: | :--- |
| `uname -a` | `0` | `Linux Rohit 6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64 GNU/Linux` |
| `uname -m` | `0` | `x86_64` |
| `cat /etc/os-release` | `0` | Ubuntu 26.04 LTS (Resolute Raccoon) |
| `free -h` | `0` | Total RAM: 3.5 GiB, Used: 558 MiB, Free: 2.3 GiB, Available: 3.0 GiB |
| `/sbin/swapon --show` | `0` | Partition `/dev/sdc`, Size: 1.0 GiB, Used: 0 B |
| `df -h` | `0` | `/dev/sdd` mounted on `/`: 1007 GiB virtual capacity, 5.1 GiB used. Host physical drive C: has 94.11 GB actual free SSD space. |
| `nproc` | `0` | 12 logical processor threads |
| `which nvidia-smi` | `0` | Located at `/usr/lib/wsl/lib/nvidia-smi` |
| `nvidia-smi` | `0` | NVIDIA GeForce GTX 1650, Driver: 581.83, CUDA capability: 13.0, 4096 MiB VRAM |
| `ls -la /dev/dxg` | `0` | `crw-rw-rw- 1 root root 10, 258` (DirectX GPU bridge device is present and accessible) |
| `df -T /home/rohit/projects` | `0` | Filesystem: `ext4` on `/dev/sdd` |

---

## 4. Hardware, Driver & Virtualization Verifications

1. **WSL2 & Architecture Confirmation**:
   - Confirmed WSL2 hypervisor architecture with Linux kernel `6.18.33.2-2`.
   - Native `x86_64` architecture confirmed both on Windows host and Linux guest.
2. **Distro Usability**:
   - Confirmed `Ubuntu 26.04 LTS` starts cleanly and runs standard Linux commands with zero errors.
3. **GPU & Driver Capability Confirmation**:
   - Windows host has NVIDIA Game Ready / Studio Driver `581.83` installed.
   - WSL2 exposes `/dev/dxg` device node and `/usr/lib/wsl/lib/nvidia-smi`.
   - NVIDIA GeForce GTX 1650 with 4096 MiB VRAM is detected in both host and WSL.
   - **Crucial Distinction**: The CUDA version "13.0" reported by `nvidia-smi` indicates driver compatibility headroom, **not** proof that the CUDA toolkit or any GPU runtime libraries are installed or operational. Prompt 02 will perform actual GPU compute verification.
4. **Physical Disk vs Virtual VHD Space**:
   - The WSL2 virtual disk (`ext4.vhdx`) is stored at `C:\Users\rohit\AppData\Local\wsl\{c1c5033f-f3df-4c61-8242-581d9c01e245}\ext4.vhdx` (current size on disk: ~5.86 GB).
   - Although `df -h /` inside WSL reports a dynamic virtual maximum of 1007 GiB, the actual physical capacity limit is the host SSD **C:**, which has **94.11 GB free space**. All intermediate reconstructions and datasets must respect this physical limit.
5. **Display & GUI Prerequisite Policy**:
   - Per project build contract, WSLg and Pangolin are **not** application prerequisites. All reconstruction processing will be headless Linux within WSL2, and visualizations will be rendered via a lightweight Windows browser viewer.

---

## 5. Measured Resource Proposal

- **Physical Host RAM**: 7.88 GB (~7.35 GiB usable).
- **WSL Allocation**: Default WSL2 auto-allocation assigns ~50% of host RAM to WSL, providing **3.5 GiB RAM** and **1.0 GiB swap**.
- **Windows Host Budget**: Retains **~3.85 GiB RAM** for Windows 11 system tasks, Brave browser, and background OS operations.
- **Recommendation**:
  - **Preserve existing settings without modifying `$env:USERPROFILE\.wslconfig`**.
  - Allocating more RAM to WSL (e.g. 5+ GB) on an 8 GB host would starve Windows 11, resulting in aggressive OS paging, UI latency, or driver stability issues.
  - 3.5 GiB RAM in WSL with 1.0 GiB swap is sufficient for headless feature extraction and pose estimation using small/medium batch sizes in PyCOLMAP. If heavy peak operations encounter memory pressure in later stages, swap can be gently increased via `.wslconfig` (e.g. `swap=2GB`) while keeping RAM capped at 3.5GB.
  - Zero modifications to existing WSL configuration files were made during this prompt.

---

## 6. Write Access & Operational Tests
- **WSL Project Directory Test**: Created directory `/home/rohit/projects/insta360_mapper/tmp_write_test`, wrote file `test.txt`, read contents (`write_test_ok`), and cleaned up. Test succeeded with exit code 0.
- **Windows Project Directory Test**: Wrote file `test_write.tmp` in Windows repository root, verified contents (`write_test_ok`), and cleaned up. Test succeeded with exit code 0.
- **Distro Lifecycle**: Distro starts cleanly on demand without errors.

---

## 7. Status Separation

| Dimension | Current Status | Notes |
| :--- | :--- | :--- |
| **Software Status** | Stage 01 Preflight Complete | Base OS, WSL2, and GPU detection verified. Headless PyCOLMAP, Python virtual environment, and CUDA runtime libraries are not yet installed (scheduled for Prompt 02). |
| **Actual-Data Status** | Input Datasets Available Locally | Raw data available in local directory (`data/2026-08-31/` and `data/2026-09-10/`). Raw files remain untouched. |
| **Scale Status** | Metric Calibration Not Initialized | Raw IMU and video streams not yet ingested for v2; metric IMU scaling branch pending later pipeline stages. |

---

## 8. Run / Stop / Resume Instructions
- **Run**: All probes completed. No persistent background daemon was started or left running.
- **Stop**: WSL instance was started for diagnostics and can idle or be gracefully shut down using `wsl.exe --terminate Ubuntu` without affecting host stability.
- **Resume**: Subsequent prompts execute directly in Windows PowerShell targeting WSL via `wsl.exe -d Ubuntu -e <command>` or within the WSL terminal at `/home/rohit/projects/insta360_mapper`.

---

## 9. Final Verdict & Readiness

**Preflight Assessment**: `HOST_PASS`

- Evidence: All required diagnostics exited with code 0. WSL2 Ubuntu 26.04 LTS is operational, NVIDIA driver 581.83 exposes the GeForce GTX 1650 via `/dev/dxg`, 94.11 GB of SSD storage is available on C:, and read/write permissions are confirmed.
- Can Prompt 02 run? **YES**.
- Limitation note: No GPU runtime computation or reconstruction has been called or tested in Prompt 01. Prompt 02 will install dependencies and test actual GPU computation.
