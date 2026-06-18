def process_specification(spec_data: dict) -> dict:
    assert "pattern" in spec_data, "specification must contain key 'pattern'"
    assert "panels" in spec_data["pattern"], (
        "specification['pattern'] must contain key 'panels'"
    )
    assert "stitches" in spec_data["pattern"], (
        "specification['pattern'] must contain key 'stitches'"
    )

    if "panel_order" not in spec_data["pattern"]:
        spec_data["pattern"]["panel_order"] = list(
            spec_data["pattern"]["panels"].keys()
        )

    if "parameters" not in spec_data:
        spec_data["parameters"] = {}

    if "parameter_order" not in spec_data:
        spec_data["parameter_order"] = []

    if "properties" not in spec_data:
        spec_data["properties"] = {
            "curvature_coords": "relative",
            "normalize_panel_translation": False,
            "normalized_edge_loops": True,
            "units_in_meter": 100,
        }

    return spec_data
