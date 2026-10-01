import pycolmap

print("CameraMode members:", [m for m in dir(pycolmap.CameraMode) if not m.startswith("_")])
for m in [pycolmap.CameraMode.AUTO, pycolmap.CameraMode.SINGLE, pycolmap.CameraMode.PER_FOLDER, pycolmap.CameraMode.PER_IMAGE]:
    print(m.name, m.value)
