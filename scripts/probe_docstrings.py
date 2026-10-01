import pycolmap

for name in ["extract_features", "match_sequential", "match_exhaustive", "match_spatial", "verify_matches", "incremental_mapping", "apply_rig_config", "read_rig_config"]:
    fn = getattr(pycolmap, name, None)
    if fn:
        print(f"=== DOC: {name} ===")
        print(fn.__doc__)
