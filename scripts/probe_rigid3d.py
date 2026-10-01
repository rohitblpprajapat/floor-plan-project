import pycolmap
import numpy as np

r = pycolmap.Rigid3d()
print("Rigid3d attributes:")
print([m for m in dir(r) if not m.startswith("_")])

# Test inverse
inv = r.inverse()
print("Inverse:", inv)
print("Rotation matrix:", r.rotation.matrix())
print("Translation:", r.translation)
print("Inverse rotation matrix:", inv.rotation.matrix())
print("Inverse translation:", inv.translation)
