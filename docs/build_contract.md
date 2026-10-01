# Build Contract: Local Insta360 Reconstruction & Annotation Application (v2)

This contract defines the immutable development and operational rules for Version 2 of the project on this HP Pavilion Gaming laptop (Windows 11 + WSL2 + NVIDIA GeForce GTX 1650).

## 1. Scope & Execution Rules
- **Prompt Isolation**: Implement only the currently supplied numbered prompt, then stop. Never jump ahead to future prompts.
- **Data Locality**: All new processing uses the local recording set; no cloud upload or remote data transmission.
- **Asset Immutability**: Keep original raw video, sensor, and source files strictly unchanged; ignore raw footage and heavy intermediates in git.
- **Contract & Handoff Review**: Read this contract and the previous stage handoff report at every subsequent stage.

## 2. Technical Stack & Architecture
- **Reconstruction Core**: Use headless PyCOLMAP, a conditional camera-IMU metric branch, supported surfaces, and persistent source-image observations for annotations.
- **Viewer Architecture**: Use a lightweight Windows browser viewer and headless Linux processing in WSL2. Do not make WSLg or Pangolin an application prerequisite.
- **Zero Heavy Stack Overhead**: Do not add Docker Desktop, Kubernetes, ROS, neural model training, or another reconstruction architecture as a default dependency.
- **Driver & WSL Integrity**: Do not install Linux NVIDIA kernel or display drivers inside WSL (rely on Windows host driver + WSL DirectX paravirtualization `/dev/dxg`), do not unregister distributions, and do not overwrite existing WSL settings.

## 3. Reliability, Safety & Resource Management
- **Resource Limits**: Respect strict host memory constraints (8 GB host RAM: preserve safe 3.5 GiB WSL allocation, 1-2 GiB swap, keeping ~3.85 GiB for Windows).
- **Process Management**: Use scoped job cancellation, atomic output completion, explicit versions, and resumable stages. Never kill unrelated host or WSL processes.
- **Status Partitioning**: Rigorously separate software status, actual-data status, and scale status in every report.

## 4. Mandatory Handoff Deliverables
Every prompt handoff must explicitly document:
1. Files changed / created.
2. Tested commands and their exact results (with exit codes).
3. Measured resources (host RAM, WSL RAM/swap, VRAM, disk space).
4. Known failures or limitations.
5. Run, stop, and resume commands.
6. Next-stage operating mode and prerequisites.
