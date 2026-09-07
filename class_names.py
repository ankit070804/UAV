# =====================================================
# class_names.py
#
# Maps segmentation class IDs (as produced by
# semantic_extractor.py, e.g. "class_28_percent") to
# human-readable labels.
#
# IMPORTANT: TartanAir's segmentation IDs are scene-specific
# and this repo never loaded the official TartanAir label
# legend for "ArchVizTinyHouseDay" — the previous code had
# THREE different, mutually contradicting guesses at this
# mapping spread across reasoning_engine.py, uav_xai_system.py,
# and semantic_labels.py (e.g. class 64 was called "Floor" in
# one file and "Cabinet" in another). That made the generated
# explanations unreliable.
#
# This file is now the ONLY place the mapping is defined.
# Treat these names as best-effort placeholders until they are
# verified against TartanAir's actual segmentation legend for
# this environment — do not treat them as ground truth.
# =====================================================

CLASS_NAMES = {
    "class_8_percent": "Chair",
    "class_27_percent": "Cabinet",
    "class_28_percent": "Furniture",
    "class_59_percent": "Curtain",
    "class_64_percent": "Floor",
    "class_70_percent": "Door",
    "class_74_percent": "Desk",
    "class_87_percent": "Window",
    "class_115_percent": "Ceiling",
    "class_119_percent": "Monitor",
    "class_123_percent": "Sofa",
    "class_125_percent": "Picture",
    "class_132_percent": "Plant",
    "class_143_percent": "Table",
    "class_146_percent": "Bed",
    "class_157_percent": "Wall",
    "class_175_percent": "Door Frame",
    "class_180_percent": "Carpet",
    "class_199_percent": "Books",
    "class_205_percent": "Lamp",
    "class_208_percent": "Computer",
    "class_214_percent": "Television",
    "class_229_percent": "Shelf",
    "class_239_percent": "Miscellaneous",
}


def get_class_name(feature_column):
    """
    feature_column looks like 'class_28_percent'.
    Falls back to a readable version of the raw column name
    (e.g. 'class_31 percent') if the ID hasn't been mapped yet,
    instead of silently mislabeling it.
    """
    return CLASS_NAMES.get(
        feature_column,
        feature_column.replace("_percent", "").replace("_", " "),
    )
