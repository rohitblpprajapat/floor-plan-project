import inspect
import pycolmap

print("=== PyCOLMAP Inspection ===")
print("Version:", getattr(pycolmap, "__version__", None))
print("has_cuda:", pycolmap.has_cuda)
if hasattr(pycolmap, "get_num_cuda_devices"):
    print("num_cuda_devices:", pycolmap.get_num_cuda_devices())

check_funcs = [
    "extract_features", "match_exhaustive", "match_sequential",
    "match_spatial", "match_custom", "verify_matches",
    "incremental_mapping", "bundle_adjustment", "triangulate_points",
    "read_rig_config", "apply_rig_config", "align_reconstruction_to_locations"
]
for f in check_funcs:
    print(f"func {f}:", hasattr(pycolmap, f))

check_classes = [
    "SiftExtractionOptions", "SiftMatchingOptions", "IncrementalPipelineOptions",
    "IncrementalMapperOptions", "TwoViewGeometryOptions", "RigConfig",
    "Database", "Reconstruction", "Camera", "Image", "Point3D", "Track", "TrackElement",
    "Rigid3d", "Sim3d", "Device"
]
for c in check_classes:
    print(f"class {c}:", hasattr(pycolmap, c))

print("\nDevice attrs:", dir(pycolmap.Device) if hasattr(pycolmap, "Device") else "None")

# Inspect extract_features signature and options
if hasattr(pycolmap, "extract_features"):
    try:
        sig = inspect.signature(pycolmap.extract_features)
        print("extract_features signature:", sig)
    except Exception as e:
        print("extract_features sig error:", e)

# Inspect match_sequential signature
if hasattr(pycolmap, "match_sequential"):
    try:
        sig = inspect.signature(pycolmap.match_sequential)
        print("match_sequential signature:", sig)
    except Exception as e:
        print("match_sequential sig error:", e)

# Inspect incremental_mapping signature
if hasattr(pycolmap, "incremental_mapping"):
    try:
        sig = inspect.signature(pycolmap.incremental_mapping)
        print("incremental_mapping signature:", sig)
    except Exception as e:
        print("incremental_mapping sig error:", e)

# Inspect match_custom
if hasattr(pycolmap, "match_custom"):
    try:
        sig = inspect.signature(pycolmap.match_custom)
        print("match_custom signature:", sig)
    except Exception as e:
        print("match_custom sig error:", e)

# Inspect Reconstruction methods
if hasattr(pycolmap, "Reconstruction"):
    recon_methods = [m for m in dir(pycolmap.Reconstruction) if not m.startswith("_")]
    print("\nReconstruction methods:", recon_methods)

# Inspect SiftExtractionOptions fields
if hasattr(pycolmap, "SiftExtractionOptions"):
    opts = pycolmap.SiftExtractionOptions()
    opt_fields = [f for f in dir(opts) if not f.startswith("_")]
    print("\nSiftExtractionOptions fields:", opt_fields)

# Inspect SiftMatchingOptions fields
if hasattr(pycolmap, "SiftMatchingOptions"):
    opts = pycolmap.SiftMatchingOptions()
    opt_fields = [f for f in dir(opts) if not f.startswith("_")]
    print("\nSiftMatchingOptions fields:", opt_fields)

# Inspect IncrementalPipelineOptions fields
if hasattr(pycolmap, "IncrementalPipelineOptions"):
    opts = pycolmap.IncrementalPipelineOptions()
    opt_fields = [f for f in dir(opts) if not f.startswith("_")]
    print("\nIncrementalPipelineOptions fields:", opt_fields)
