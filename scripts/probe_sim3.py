import pycolmap
import numpy as np

src = np.array([
    [0.0, 0.0, -1.0],
    [1.0, 0.0, -0.6],
    [0.0, 1.0, -0.2],
    [-1.0, 0.0, 0.2],
    [0.0, -1.0, 0.6],
], dtype=np.float64)

scale_gt = 2.5
R_gt = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])
t_gt = np.array([1.0, 2.0, 3.0])

dst = (scale_gt * (src @ R_gt.T)) + t_gt

sim3 = pycolmap.estimate_sim3d(src, dst)
print("Estimated Sim3:", sim3)
print("Scale:", sim3.scale)
print("Rot diff:", np.max(np.abs(sim3.rotation.matrix() - R_gt)))
print("Trans diff:", np.max(np.abs(sim3.translation - t_gt)))
