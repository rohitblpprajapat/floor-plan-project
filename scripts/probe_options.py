import pycolmap

def inspect_opt(cls_name):
    cls = getattr(pycolmap, cls_name, None)
    if cls is None:
        # maybe under _core
        core = getattr(pycolmap, "_core", None)
        cls = getattr(core, cls_name, None)
    if cls:
        inst = cls()
        fields = [f for f in dir(inst) if not f.startswith("_")]
        print(f"\n=== {cls_name} ===")
        print(fields)
        for f in fields:
            val = getattr(inst, f)
            if not callable(val):
                print(f"  {f}: {val}")

inspect_opt("ImageReaderOptions")
inspect_opt("FeatureExtractionOptions")
inspect_opt("FeatureMatchingOptions")
inspect_opt("SequentialPairingOptions")
inspect_opt("TwoViewGeometryOptions")
inspect_opt("CameraMode")
