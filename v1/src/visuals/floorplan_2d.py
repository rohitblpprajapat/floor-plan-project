"""
2D Floor Plan & Multi-View Orthographic Visualizer.
Transforms gravity-aligned metric trajectories and 3D landmarks into:
1. High-contrast architectural top-down 2D floor plans with wall density contours.
2. Multi-view orthographic technical projections (Top X-Y, Front X-Z, Side Y-Z).
3. Motion, elevation, and speed profile charts.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, Tuple, List
from src.stella.stella_io import VisualTrajectory

class FloorPlan2DVisualizer:
    """Renders 2D floor plans, wall density maps, and technical orthographic projections."""

    def __init__(self, dpi: int = 200):
        self.dpi = dpi

    def render_topdown_floorplan(
        self,
        trajectory: VisualTrajectory,
        landmarks: np.ndarray,
        output_path: str,
        title: str = "Metric 2D Floor Plan & Trajectory",
        start_metadata: Optional[Dict] = None
    ) -> str:
        """
        Renders an architectural top-down 2D floor plan projection.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pts_traj = trajectory.positions # (N, 3)
        ts = trajectory.timestamps

        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        fig.patch.set_facecolor("#1e1e24")
        ax.set_facecolor("#121216")

        # 1. 2D Landmark Density / Wall Boundaries
        if len(landmarks) > 50:
            lx = landmarks[:, 0]
            ly = landmarks[:, 1]
            lz = landmarks[:, 2]

            # Filter outlier points outside 3-sigma
            mx, sx = np.median(lx), np.std(lx)
            my, sy = np.median(ly), np.std(ly)
            mask = (np.abs(lx - mx) < 3.5 * sx) & (np.abs(ly - my) < 3.5 * sy)
            lx_filt, ly_filt = lx[mask], ly[mask]

            # Compute 2D histogram grid for density
            nbins = 120
            h_grid, xedges, yedges = np.histogram2d(lx_filt, ly_filt, bins=nbins)
            h_smooth = gaussian_filter(h_grid, sigma=1.2)

            # Custom glowing cyan colormap for wall density
            cmap_density = LinearSegmentedColormap.from_list(
                "wall_glow",
                [(0.0, "#121216"), (0.2, "#003b46"), (0.6, "#007577"), (1.0, "#00e5ff")]
            )


            ax.imshow(
                h_smooth.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin="lower",
                cmap=cmap_density,
                aspect="equal",
                alpha=0.85
            )

            # Scatter sparse high-confidence landmarks
            ax.scatter(lx_filt, ly_filt, s=2, c="#4dd0e1", alpha=0.35, label="3D Surface Landmarks")

        # 2. Camera Trajectory Polyline
        x_tr = pts_traj[:, 0]
        y_tr = pts_traj[:, 1]

        # Draw trajectory with gradient segment colors
        num_segs = len(x_tr) - 1
        cmap_traj = plt.get_cmap("spring")
        for i in range(num_segs):
            color = cmap_traj(i / max(num_segs, 1))
            ax.plot(x_tr[i : i + 2], y_tr[i : i + 2], color=color, linewidth=2.5, solid_capstyle="round", zorder=4)

        # 3. Direction of Travel Arrows
        step_arrow = max(1, len(x_tr) // 12)
        for i in range(0, len(x_tr) - 1, step_arrow):
            dx = x_tr[i + 1] - x_tr[i]
            dy = y_tr[i + 1] - y_tr[i]
            norm = np.hypot(dx, dy)
            if norm > 1e-4:
                ax.annotate(
                    "",
                    xy=(x_tr[i] + dx * 0.7, y_tr[i] + dy * 0.7),
                    xytext=(x_tr[i], y_tr[i]),
                    arrowprops=dict(arrowstyle="->", color="#ffff00", lw=1.5),
                    zorder=5
                )

        # 4. Start and End Markers
        ax.scatter([x_tr[0]], [y_tr[0]], s=140, c="#00ff66", edgecolors="white", linewidth=2, zorder=6, label="Start Point (t=0s)")
        ax.scatter([x_tr[-1]], [y_tr[-1]], s=140, c="#ff0055", edgecolors="white", linewidth=2, zorder=6, label=f"End Point (t={ts[-1]:.1f}s)")

        # Start coordinates note if metadata available
        if start_metadata:
            bld = start_metadata.get("building", {}).get("name", "Survey")
            flr = start_metadata.get("floor", {}).get("label", "Floor 1")
            ax.text(
                x_tr[0] + 0.03, y_tr[0] - 0.03,
                f"Start: {bld} - {flr}",
                color="#00ff66", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a24", edgecolor="#00ff66", alpha=0.8)
            )

        # 5. Metric Grid & Ruler Scale Bar
        ax.grid(True, color="#2f313d", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_aspect("equal")

        # Compute span and scale bar
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        span_x = x_max - x_min
        span_y = y_max - y_min

        # 1-meter or 0.5-meter scale bar
        scale_len = 0.5 if span_x < 2.0 else 1.0
        bar_x0 = x_min + 0.08 * span_x
        bar_y0 = y_min + 0.06 * span_y
        ax.plot([bar_x0, bar_x0 + scale_len], [bar_y0, bar_y0], color="white", linewidth=4, zorder=7)
        ax.text(
            bar_x0 + scale_len / 2.0, bar_y0 + 0.02 * span_y,
            f"{scale_len:.1f} m", color="white", fontsize=10, fontweight="bold",
            ha="center", va="bottom", zorder=7
        )

        # Bounding box dimensions
        dx_m = float(np.max(x_tr) - np.min(x_tr))
        dy_m = float(np.max(y_tr) - np.min(y_tr))
        path_len = float(np.sum(np.linalg.norm(np.diff(pts_traj, axis=0), axis=1)))

        info_text = (
            f"Trajectory Path: {path_len:.2f} m\n"
            f"Bounding Box: {dx_m:.2f}m x {dy_m:.2f}m\n"
            f"Duration: {ts[-1] - ts[0]:.1f} s | Keyframes: {len(trajectory)}"
        )
        ax.text(
            0.98, 0.98, info_text,
            transform=ax.transAxes, color="white", fontsize=9.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e28", edgecolor="#555566", alpha=0.85)
        )

        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("X (meters)", color="#c0c0c0", fontsize=11)
        ax.set_ylabel("Y (meters)", color="#c0c0c0", fontsize=11)
        ax.tick_params(colors="#a0a0a0")
        for spine in ax.spines.values():
            spine.set_color("#3a3a46")

        ax.legend(loc="lower right", facecolor="#1e1e28", edgecolor="#444455", labelcolor="white", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=self.dpi)
        plt.close()
        return output_path

    def render_multiview_orthographic(
        self,
        trajectory: VisualTrajectory,
        landmarks: np.ndarray,
        output_path: str
    ) -> str:
        """
        Renders a 3-view orthographic technical projection (Top X-Y, Front X-Z, Side Y-Z).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pts_traj = trajectory.positions

        fig = plt.figure(figsize=(15, 5), dpi=self.dpi)
        fig.patch.set_facecolor("#18181f")

        # Top View (X-Y)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_facecolor("#0e0e14")
        if len(landmarks) > 0:
            ax1.scatter(landmarks[:, 0], landmarks[:, 1], s=1, c="#00bcd4", alpha=0.3)
        ax1.plot(pts_traj[:, 0], pts_traj[:, 1], c="#ff007f", linewidth=2, label="Camera Path")
        ax1.scatter(pts_traj[0, 0], pts_traj[0, 1], s=70, c="#00ff66", label="Start")
        ax1.scatter(pts_traj[-1, 0], pts_traj[-1, 1], s=70, c="#ff0033", label="End")
        ax1.set_title("Top View (Floor Plan X-Y)", color="white", fontsize=11, fontweight="bold")
        ax1.set_xlabel("X (m)", color="#b0b0b0")
        ax1.set_ylabel("Y (m)", color="#b0b0b0")
        ax1.set_aspect("equal")
        ax1.grid(True, color="#252530", linestyle="--")

        # Front View (X-Z)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_facecolor("#0e0e14")
        if len(landmarks) > 0:
            ax2.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c="#00bcd4", alpha=0.3)
        ax2.plot(pts_traj[:, 0], pts_traj[:, 2], c="#ff007f", linewidth=2)
        ax2.scatter(pts_traj[0, 0], pts_traj[0, 2], s=70, c="#00ff66")
        ax2.scatter(pts_traj[-1, 0], pts_traj[-1, 2], s=70, c="#ff0033")
        ax2.set_title("Front Elevation View (X-Z)", color="white", fontsize=11, fontweight="bold")
        ax2.set_xlabel("X (m)", color="#b0b0b0")
        ax2.set_ylabel("Z (Height, m)", color="#b0b0b0")
        ax2.grid(True, color="#252530", linestyle="--")

        # Side View (Y-Z)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_facecolor("#0e0e14")
        if len(landmarks) > 0:
            ax3.scatter(landmarks[:, 1], landmarks[:, 2], s=1, c="#00bcd4", alpha=0.3)
        ax3.plot(pts_traj[:, 1], pts_traj[:, 2], c="#ff007f", linewidth=2)
        ax3.scatter(pts_traj[0, 1], pts_traj[0, 2], s=70, c="#00ff66")
        ax3.scatter(pts_traj[-1, 1], pts_traj[-1, 2], s=70, c="#ff0033")
        ax3.set_title("Side Elevation View (Y-Z)", color="white", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Y (m)", color="#b0b0b0")
        ax3.set_ylabel("Z (Height, m)", color="#b0b0b0")
        ax3.grid(True, color="#252530", linestyle="--")

        for ax in [ax1, ax2, ax3]:
            ax.tick_params(colors="#909090")
            for spine in ax.spines.values():
                spine.set_color("#303040")

        ax1.legend(facecolor="#181824", edgecolor="#3a3a4a", labelcolor="white", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=self.dpi)
        plt.close()
        return output_path

    def render_motion_profiles(
        self,
        trajectory: VisualTrajectory,
        output_path: str
    ) -> str:
        """
        Renders speed, elevation profile, and cumulative path length over time.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pts = trajectory.positions
        ts = trajectory.timestamps

        diffs = np.diff(pts, axis=0)
        dt = np.diff(ts)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(dists)])
        speeds = np.concatenate([[0.0], dists / np.maximum(dt, 1e-4)])

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=self.dpi, sharex=True)
        fig.patch.set_facecolor("#18181f")

        # 1. Cumulative Distance
        axes[0].set_facecolor("#0e0e14")
        axes[0].plot(ts, cum_dist, color="#00e5ff", linewidth=2.2)
        axes[0].fill_between(ts, cum_dist, color="#00e5ff", alpha=0.15)
        axes[0].set_ylabel("Distance (m)", color="white", fontsize=10)
        axes[0].set_title("Surveying Motion & Telemetry Profiles", color="white", fontsize=13, fontweight="bold")

        # 2. Linear Speed
        axes[1].set_facecolor("#0e0e14")
        axes[1].plot(ts, speeds, color="#ff9100", linewidth=1.8)
        axes[1].fill_between(ts, speeds, color="#ff9100", alpha=0.15)
        axes[1].set_ylabel("Speed (m/s)", color="white", fontsize=10)

        # 3. Z Elevation
        axes[2].set_facecolor("#0e0e14")
        axes[2].plot(ts, pts[:, 2], color="#d500f9", linewidth=2.0)
        axes[2].fill_between(ts, pts[:, 2], color="#d500f9", alpha=0.15)
        axes[2].set_ylabel("Height Z (m)", color="white", fontsize=10)
        axes[2].set_xlabel("Time (seconds)", color="white", fontsize=10)

        for ax in axes:
            ax.grid(True, color="#252530", linestyle="--")
            ax.tick_params(colors="#909090")
            for spine in ax.spines.values():
                spine.set_color("#303040")

        plt.tight_layout()
        plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=self.dpi)
        plt.close()
        return output_path
