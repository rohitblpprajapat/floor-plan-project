import pycolmap

rec = pycolmap.Reconstruction()
print("Reconstruction attributes and methods:")
attrs = [m for m in dir(rec) if not m.startswith("_")]
print(attrs)

img = pycolmap.Image()
print("\nImage attributes:")
print([m for m in dir(img) if not m.startswith("_")])

pt = pycolmap.Point3D()
print("\nPoint3D attributes:")
print([m for m in dir(pt) if not m.startswith("_")])
