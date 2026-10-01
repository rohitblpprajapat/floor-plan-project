# Projection Adapter Documentation: Equirectangular to Virtual Pinhole Rig

This document specifies the mathematics, coordinate systems, pixel conventions, seam wrapping, and PyCOLMAP multi-camera rig serialization implemented in `vscan.frames`.

---

## 1. Coordinate Systems & Conventions

### 1.1 Camera Optical Frame (OpenCV Standard)
Every perspective camera uses the standard right-handed optical coordinate frame:
- **+X**: Points to the **right** along image rows.
- **+Y**: Points **down** along image columns.
- **+Z**: Points **forward** along the optical axis into the scene.

### 1.2 Panorama Coordinate Frame
The equirectangular panorama body frame is defined synchronously with the camera rig root:
- **+X**: Right ($90^\circ$ azimuth, East).
- **+Y**: Down (Nadir, gravity down direction when unpitched).
- **+Z**: Forward ($0^\circ$ azimuth, North).

Spherical coordinates $(\theta, \phi)$ correspond to:
- **Azimuth $\theta \in [-\pi, \pi]$**: Rotation around $+Y$.
  - $\theta = 0$: Forward (+Z)
  - $\theta = +\pi/2$: Right (+X)
  - $\theta = -\pi/2$: Left (-X)
  - $\theta = \pm \pi$: Backward (-Z, horizontal seam boundary)
- **Elevation $\phi \in [-\pi/2, \pi/2]$**: Angle from horizontal plane ($Y=0$).
  - $\phi = +\pi/2$: Zenith (pointing $-Y$, straight up)
  - $\phi = 0$: Equator (horizontal horizon plane)
  - $\phi = -\pi/2$: Nadir (pointing $+Y$, straight down)

The 3D unit bearing vector $\vec{d}_{pano} = [x, y, z]^T$ in the panorama frame is:
$$\begin{aligned}
x &= \cos(\phi) \sin(\theta) \\
y &= -\sin(\phi) \\
z &= \cos(\phi) \cos(\theta)
\end{aligned}$$
with exact inverse:
$$\begin{aligned}
\phi &= -\arcsin(\operatorname{clip}(y, -1.0, 1.0)) \\
\theta &= \operatorname{atan2}(x, z)
\end{aligned}$$

---

## 2. Documented Pixel-Center Convention

In all projection models within `vscan`:
- Pixel coordinates $(i, j)$ are integers representing discrete sensor elements: $0 \le i < W$, $0 \le j < H$.
- The **sampling center** of pixel $(i, j)$ is defined at the **half-integer offset**:
  $$(u_c, v_c) = (i + 0.5, j + 0.5)$$
- In continuous coordinate space, pixel $(i, j)$ covers the continuous box $[i, i + 1) \times [j, j + 1)$.
- On an equirectangular image of dimensions $W_{eq} \times H_{eq}$:
  $$\theta = \left(\frac{u_c}{W_{eq}} - 0.5\right) \cdot 2\pi$$
  $$\phi = \left(0.5 - \frac{v_c}{H_{eq}}\right) \cdot \pi$$
- This guarantees that $\theta = 0$ corresponds to the exact horizontal center $u_c = W_{eq} / 2$, and $\phi = 0$ corresponds to the vertical center $v_c = H_{eq} / 2$.

---

## 3. Virtual Pinhole Intrinsics

Each virtual perspective view is rendered with exact pinhole camera intrinsics:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

For a square virtual view with field of view $\text{FOV} = 90^\circ$ and resolution $W \times H$:
$$\alpha = \frac{\text{FOV}}{2} = 45^\circ = \frac{\pi}{4}$$
$$f_x = f_y = \frac{W / 2}{\tan(\alpha)} = \frac{W}{2}$$
$$c_x = \frac{W}{2.0}, \quad c_y = \frac{H}{2.0}$$

For the standard $960 \times 960$ virtual face:
- $f_x = 480.0$, $f_y = 480.0$
- $c_x = 480.0$, $c_y = 480.0$

Under image scaling by $(s_x, s_y)$:
$$f'_x = s_x f_x, \quad f'_y = s_y f_y, \quad c'_x = s_x c_x, \quad c'_y = s_y c_y$$
Under image cropping by $(x_0, y_0, w_{crop}, h_{crop})$:
$$f'_x = f_x, \quad f'_y = f_y, \quad c'_x = c_x - x_0, \quad c'_y = c_y - y_0$$
The reconstructed bearing rays $\vec{d}_c$ remain identical across crops and rescales.

---

## 4. The 6-Face Cubemap Rig

The virtual rig consists of 6 cameras positioned with **exactly zero synthetic baseline**:
$$\vec{t}_{pano\_cam} = [0.0, 0.0, 0.0]^T \quad \text{for all } 6 \text{ faces}$$

The 6 rotation matrices $R_{pano\_cam}$ transform bearing rays from the camera optical frame into the panorama body frame ($\vec{d}_{pano} = R_{pano\_cam} \vec{d}_c$):

1. **`face_front` (Reference sensor, cam0)**:
   $$R_{pano\_cam} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad \text{Optical Axis: } +Z$$

2. **`face_right` (cam1, Yaw $+90^\circ$)**:
   $$R_{pano\_cam} = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}, \quad \text{Optical Axis: } +X$$

3. **`face_back` (cam2, Yaw $180^\circ$)**:
   $$R_{pano\_cam} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad \text{Optical Axis: } -Z$$

4. **`face_left` (cam3, Yaw $-90^\circ$)**:
   $$R_{pano\_cam} = \begin{bmatrix} 0 & 0 & -1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}, \quad \text{Optical Axis: } -X$$

5. **`face_up` (cam4, Pitch $-90^\circ$)**:
   $$R_{pano\_cam} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}, \quad \text{Optical Axis: } -Y \text{ (Zenith)}$$

6. **`face_down` (cam5, Pitch $+90^\circ$)**:
   $$R_{pano\_cam} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \text{Optical Axis: } +Y \text{ (Nadir)}$$

Every matrix is strictly proper SO(3) with $\det(R) = +1.0$ and $R^T R = I$.

---

## 5. Seam Wrapping and Polar Behavior

1. **Horizontal Seam Wrapping**:
   Equirectangular azimuth $\theta$ wraps around $[-\pi, \pi]$ with period $2\pi$.
   During vector reprojection, map coordinates $u_{map}$ are transformed with modular arithmetic:
   $$u_{map} = \left(\left(\frac{\theta}{2\pi} + 0.5\right) W_{eq} - 0.5\right) \pmod{W_{eq}}$$
   This ensures that the horizontal seam across $\theta = \pm \pi$ is continuous, and any object or polygon spanning the seam is rendered as a contiguous unbroken region in `face_back`.

2. **Polar Boundary Clamping**:
   At elevation extremes $\phi \to \pm \pi/2$ (zenith and nadir), vertical coordinates $v_{map}$ are bounded:
   $$v_{map} = \operatorname{clip}\left(\left(0.5 - \frac{\phi}{\pi}\right) H_{eq} - 0.5, \, 0.0, \, H_{eq} - 1.0\right)$$
   This guarantees zero numerical instability or coordinate bleeding at the poles.

---

## 6. PyCOLMAP 4.2.0 RigConfig Serialization

In PyCOLMAP 4.2.0, multi-camera rigs are specified via JSON:
```json
[
  {
    "cameras": [
      {
        "image_prefix": "face_front_",
        "ref_sensor": true
      },
      {
        "image_prefix": "face_right_",
        "ref_sensor": false,
        "cam_from_rig_rotation": [0.7071067811865476, 0.0, -0.7071067811865476, 0.0],
        "cam_from_rig_translation": [0.0, 0.0, 0.0]
      },
      ...
    ]
  }
]
```
Where:
- `cam_from_rig_rotation` specifies the unit quaternion $[w, x, y, z]$ representing $R_{cam\_rig} = (R_{pano\_cam})^T$.
- `cam_from_rig_translation` is strictly $[0.0, 0.0, 0.0]$.
- When parsed by PyCOLMAP, the inverse transform $(R_{cam\_rig})^T \vec{d}_c$ reproduces the exact bearing rays of the panorama.

---

## 7. Camera Stabilization & Visual Baseline Approximations

1. **Time-Dependent Stabilization**:
   In Insta360 stitched panoramic exports, vendor FlowState stabilization applies a continuous time-dependent 3D rotation $R_{stabilization}(t)$ to maintain a level horizon.
   Because the uncalibrated vendor stabilization transform changes per frame, an unqualified fixed static rotation between the image frame and the camera IMU is blocked.
   The image-to-IMU orientation is marked `STABILIZATION_UNKNOWN_TIME_DEPENDENT`.

2. **Visual Baseline Approximation**:
   When raw dual-fisheye lens calibration parameters are unavailable, the stitched equirectangular export is used as a provisional visual baseline.
   All scale status remains strictly `NOT_ESTIMATED`.
