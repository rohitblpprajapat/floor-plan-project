"""
Interactive 3D Web Viewer Generator.
Creates a self-contained, interactive HTML 3D scene (powered by Plotly) displaying:
- 3D metric camera trajectory path
- 3D camera orientation frustums / heading vectors
- 3D metric landmarks / point cloud colored by elevation
- Interactive hover tooltips with metric coordinates
"""

import os
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict
from src.stella.stella_io import VisualTrajectory

class Interactive3DVisualizer:
    """Generates standalone interactive 3D HTML models of trajectory and point cloud."""

    @staticmethod
    def render_interactive_html(
        trajectory: VisualTrajectory,
        landmarks: np.ndarray,
        output_path: str,
        title: str = "Metric 360° Visual-Inertial 3D Reconstruction"
    ) -> str:
        """
        Builds a self-contained HTML file with full 3D interactive controls.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pts_traj = trajectory.positions
        ts = trajectory.timestamps

        fig = go.Figure()

        # 1. 3D Point Cloud (Landmarks)
        if len(landmarks) > 0:
            lx = landmarks[:, 0]
            ly = landmarks[:, 1]
            lz = landmarks[:, 2]

            # Filter outlier points outside 3.5 sigma
            mx, sx = np.median(lx), np.std(lx)
            my, sy = np.median(ly), np.std(ly)
            mz, sz = np.median(lz), np.std(lz)
            mask = (np.abs(lx - mx) < 3.5 * sx) & (np.abs(ly - my) < 3.5 * sy) & (np.abs(lz - mz) < 3.5 * sz)

            lx_filt = lx[mask]
            ly_filt = ly[mask]
            lz_filt = lz[mask]

            # Downsample if > 15,000 for browser responsiveness
            if len(lx_filt) > 15000:
                idx = np.random.choice(len(lx_filt), 15000, replace=False)
                lx_filt, ly_filt, lz_filt = lx_filt[idx], ly_filt[idx], lz_filt[idx]

            fig.add_trace(go.Scatter3d(
                x=lx_filt,
                y=ly_filt,
                z=lz_filt,
                mode="markers",
                marker=dict(
                    size=2.2,
                    color=lz_filt,
                    colorscale="Viridis",
                    opacity=0.65,
                    colorbar=dict(title="Height Z (m)", x=1.02, len=0.6)
                ),
                name="3D Surface Landmarks",
                hovertemplate="<b>Landmark</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>"
            ))

        # 2. 3D Camera Trajectory Path
        fig.add_trace(go.Scatter3d(
            x=pts_traj[:, 0],
            y=pts_traj[:, 1],
            z=pts_traj[:, 2],
            mode="lines+markers",
            line=dict(
                color="#ff007f",
                width=6
            ),
            marker=dict(
                size=3.5,
                color="#ffeb3b",
            ),
            name="Camera Trajectory",
            hovertemplate="<b>Keyframe</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<extra></extra>"
        ))

        # 3. Start and End Markers
        fig.add_trace(go.Scatter3d(
            x=[pts_traj[0, 0]],
            y=[pts_traj[0, 1]],
            z=[pts_traj[0, 2]],
            mode="markers+text",
            marker=dict(size=10, color="#00ff66", symbol="diamond"),
            text=["START (0.0s)"],
            textposition="top center",
            name="Start Point",
            hoverinfo="text"
        ))

        fig.add_trace(go.Scatter3d(
            x=[pts_traj[-1, 0]],
            y=[pts_traj[-1, 1]],
            z=[pts_traj[-1, 2]],
            mode="markers+text",
            marker=dict(size=10, color="#ff0033", symbol="diamond"),
            text=[f"END ({ts[-1]:.1f}s)"],
            textposition="top center",
            name="End Point",
            hoverinfo="text"
        ))

        # 4. Camera Viewing Frustums / Direction Vectors (sampled every 10 keyframes)
        step_kf = max(1, len(trajectory) // 15)
        for i in range(0, len(trajectory), step_kf):
            kf = trajectory.keyframes[i]
            p = kf.p_wc
            # Optical axis in Camera frame is +Z or forward
            fwd_w = kf.R_wc @ np.array([0.0, 0.0, 0.15]) # 15cm vector forward
            p_end = p + fwd_w

            fig.add_trace(go.Scatter3d(
                x=[p[0], p_end[0]],
                y=[p[1], p_end[1]],
                z=[p[2], p_end[2]],
                mode="lines",
                line=dict(color="#00e5ff", width=4),
                showlegend=(i == 0),
                name="Camera View Vector" if i == 0 else None,
                hoverinfo="none"
            ))

        # 5. Scene Layout & Dark Theme
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:12px;color:#a0a0a0'>Gravity-Aligned Metric Frame (SI Meters)</span>",
                x=0.05,
                y=0.95,
                font=dict(color="white", size=16)
            ),
            paper_bgcolor="#121216",
            scene=dict(
                bgcolor="#121216",
                xaxis=dict(title="X (meters)", color="#a0a0a0", gridcolor="#2a2a35", showbackground=False),
                yaxis=dict(title="Y (meters)", color="#a0a0a0", gridcolor="#2a2a35", showbackground=False),
                zaxis=dict(title="Z Height (meters)", color="#a0a0a0", gridcolor="#2a2a35", showbackground=False),
                aspectmode="data"
            ),
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(25, 25, 32, 0.8)",
                bordercolor="#3a3a46",
                borderwidth=1,
                x=0.02,
                y=0.1
            ),
            margin=dict(l=0, r=0, b=0, t=60)
        )

        fig.write_html(output_path, include_plotlyjs=True)
        return output_path
