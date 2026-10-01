import pycolmap

im = pycolmap.Image()
print("cam_from_world type:", type(getattr(im, "cam_from_world", None)))
