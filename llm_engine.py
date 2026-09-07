import os
import re
import ollama

# =====================================================
# MODEL
#
# Configurable via UAV_LLM_MODEL so this doesn't require
# editing source to try a different local Ollama model.
# =====================================================

MODEL_NAME = os.environ.get("UAV_LLM_MODEL", "llama3.2")

# =====================================================
# WHAT CHANGED FROM THE PREVIOUS VERSION
#
# The old version asked the LLM for one hedge sentence PER feature and
# stitched them into a bulleted "Reasons for Localization Error" list
# alongside a raw table of feature/value/SHAP numbers. That read like a
# debug dump, not an explanation a normal person could use.
#
# This version instead asks the LLM to read the top contributing
# factors (given to it, not shown to the end user) and synthesize them
# into ONE short, plain-English paragraph — the way a person would
# actually explain "why" something happened, not a table. The raw
# feature/value/SHAP numbers are only ever used internally to build the
# prompt; they are never part of the final report text.
# =====================================================

SYSTEM_PROMPT = """
You are explaining, to a non-technical UAV operator, why the system
predicted a certain amount of localization error for one video frame.

You will be given the predicted error and a short list of the factors
that most influenced that specific prediction, each with its measured
value, a magnitude label, and whether it pushed the error up or down.
This list is for YOUR understanding only — the operator will never see
it, so do not present it as a list or table.

Your job is to write exactly two things:

1. Explanation — ONE short paragraph (3-6 sentences) that reads like a
   person explaining what happened, weaving the most important factors
   together into a single coherent story of why this much error was
   predicted. Do not write "Factor 1... Factor 2..." or a bullet list.
   Use plain, everyday language a non-expert can follow — describe what
   each factor practically means (e.g. "the drone moved a noticeable
   amount between frames" instead of "dx"; "the system found very few
   reliable visual details to track" instead of "orb_features") rather
   than naming the raw feature.

2. Recommendations — 2 to 4 short, practical bullet points, grounded
   only in the factors you were given.

Hard rules:

- Only use the factors, values, and directions given to you. Never
  invent a factor, value, or direction that wasn't provided, and never
  contradict the direction given for any factor.
- Where a factor has a magnitude label (SMALL/MODERATE/LARGE/HIGH/LOW),
  that label is a pre-computed fact — never contradict it or re-judge
  the raw number yourself.
- You may propose a plausible reason WHY a factor had its effect, but
  phrase it as a hedge ("this may suggest...", "one possibility is...",
  "this could indicate..."), never as a stated fact — you were only
  given the measured value and its direction, not the true mechanism.
- The individual factor values are NOT meters and are not directly
  comparable to the overall predicted error in meters. Never write the
  word "meters" (or a number followed by it) anywhere in your response —
  the total predicted error in meters is already stated separately by
  someone else; you must never restate it or attach "meters" to any
  individual factor's value.
- Never invent a cause, mechanism, or fix that isn't implied by the
  factors you were given. Do not invent claims about UAV speed, flight
  path, weather, obstacles, camera resolution, lighting, altitude,
  orientation, hardware, or alternative algorithms/techniques — unless
  one of the given factors directly represents it.
- Never invent a system component or internal process (e.g.
  "compensation mechanism", "error model", "correction system") — you
  were only given factor values and their effect on the prediction, not
  how the system works internally.
- Never invent a physical property of an object beyond what a factor
  tells you. A factor that is a percentage of the image occupied by an
  object only tells you how much space it takes up — it does not tell
  you the object is "solid", "reflective", "cluttered", or anything
  else about its physical nature.
- Do not assert a specific real-world direction (forward, backward,
  sideways, left, right, up, down) for the UAV's movement between
  frames — the data does not specify which physical direction the
  motion axes point, only how much the UAV moved along them.
- Every recommendation must follow logically from the factors you were
  given for this frame. Do not recommend anything involving equipment,
  settings, or techniques that weren't given as a factor.
- Never mention "SHAP", "Random Forest", "machine learning", "SLAM",
  "ORB-SLAM", "SIFT", "SURF", or any specific algorithm name.
- Be concise, warm, and clear — write for someone with no technical
  background in computer vision or robotics.

Output format — return ONLY this, nothing else:

Explanation
<your one paragraph>

Recommendations
1. <first recommendation>
2. <second recommendation>
(2 to 4 total)
"""


def compute_risk_level(prediction):
    if prediction < 0.30:
        return "Low (error < 0.30 m)"
    elif prediction <= 0.80:
        return "Medium (0.30-0.80 m)"
    else:
        return "High (> 0.80 m)"


def describe_feature(raw_name):
    """
    What each raw feature column actually measures, in plain terms.
    Fed to the LLM as internal context only — never shown to the user.
    """

    if raw_name.startswith("class_") and raw_name.endswith("_percent"):
        return "percentage of the visible camera image occupied by this object/surface"
    if raw_name in ("dx", "dy", "dz"):
        axis = raw_name[-1].upper()
        return (
            f"the UAV's movement along the {axis} axis between this frame "
            f"and the next, in meters (the physical direction this axis "
            f"points - e.g. forward, sideways, up - is not specified in "
            f"this data; do not assert a specific direction)"
        )
    if raw_name == "orb_features":
        return "the number of distinct visual tracking keypoints detected in the image"
    if raw_name == "blur":
        return "an image sharpness score (higher = sharper, less motion blur)"
    if raw_name == "contrast":
        return "an image contrast score"
    if raw_name == "brightness":
        return "average image brightness (0-255 grayscale scale)"
    if raw_name == "edge_density":
        return "fraction of image pixels detected as edges"
    if raw_name in ("mean_depth", "std_depth", "min_depth", "max_depth", "depth_range", "depth_entropy", "valid_depth_ratio"):
        return "a statistic computed from the depth image (not a spatial displacement)"
    return "a computed image/motion statistic (exact unit not specified — do not guess a unit for it)"


def describe_magnitude(raw_name, value):
    """
    Qualitative label for the raw measured value, computed in Python so
    the LLM never has to eyeball (and potentially misjudge) raw numbers.
    """

    if raw_name.startswith("class_") and raw_name.endswith("_percent"):
        if value < 2:
            return "a SMALL fraction of the image"
        elif value < 10:
            return "a MODERATE fraction of the image"
        else:
            return "a LARGE fraction of the image"

    if raw_name in ("dx", "dy", "dz"):
        av = abs(value)
        if av < 0.05:
            return "a SMALL movement"
        elif av < 0.3:
            return "a MODERATE movement"
        else:
            return "a LARGE movement"

    if raw_name == "orb_features":
        if value < 1500:
            return "a LOW number of keypoints"
        elif value < 4000:
            return "a MODERATE number of keypoints"
        else:
            return "a HIGH number of keypoints"

    return None


# =====================================================
# FILTERING
#
# The LLM has, in testing, occasionally ignored a prompt rule anyway
# (e.g. asserting "forward" for dx, or recommending an "algorithm
# change" never given as a factor). Rather than trust the prompt alone,
# these filters remove offending sentences/bullets after generation.
# Anything removed is only logged to the console (for debugging) — it
# is never written into the final report text, since the whole point is
# a clean report for a non-technical reader.
# =====================================================

_UNGROUNDED_TERMS = [
    "algorithm", "processing power", "camera resolution", "lighting",
    "hardware", "threshold", "filter", "slam", "software", "firmware",
    "calibrat", "sensor fusion", "gps",
    "compensation mechanism", "error model", "correction system",
    "compensat", "over-compensat", "overcompensat",
    "reflective", "unobstructed", "clear opening", "solid object",
    "transparent", "translucent", "opaque",
]

_AXIS_PATTERN = re.compile(r"\b(dx|dy|dz|[xyz]\s*[- ]?\s*axis)\b", re.IGNORECASE)
_DIRECTION_TERMS_PATTERN = re.compile(
    r"\b(forward(s)?|backward(s)?|sideways|lateral(ly)?|leftward(s)?|"
    r"rightward(s)?|upward(s)?|downward(s)?|horizontal(ly)?|vertical(ly)?)\b",
    re.IGNORECASE,
)
# Catches the LLM mislabeling an individual factor's raw value as a
# distance (e.g. "a decrease of 0.252 meters", "4.708 meters of the
# image") — only the overall predicted error, written separately in
# Python, is actually in meters.
_MISLABELED_METERS_PATTERN = re.compile(r"\d+(\.\d+)?\s*meters?\b", re.IGNORECASE)


def _is_flagged(text):
    lower = text.lower()
    if any(term in lower for term in _UNGROUNDED_TERMS):
        return True
    if _AXIS_PATTERN.search(text) and _DIRECTION_TERMS_PATTERN.search(text):
        return True
    if _MISLABELED_METERS_PATTERN.search(text):
        return True
    return False


def _split_sentences(text):
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _filter_prose(text, label):
    """
    Sentence-level filter for the free-flowing Explanation paragraph:
    drops any sentence that uses an ungrounded term, asserts a specific
    physical direction for a motion axis, or mislabels a factor's raw
    value as a distance in meters. Logs how many sentences were dropped
    instead of leaving any trace in the report.
    """

    kept = []
    removed = 0

    for sentence in _split_sentences(text):
        if _is_flagged(sentence):
            removed += 1
            continue
        kept.append(sentence)

    if removed:
        print(
            f"[llm_engine] {label}: removed {removed} sentence(s) that "
            f"used an ungrounded term, asserted an unsupported physical "
            f"direction, or mislabeled a factor's value as meters."
        )

    return " ".join(kept).strip()


_LIST_ITEM_START_PATTERN = re.compile(r"^([-*•]|\d+[.)])\s*")


def _is_list_item_start(stripped_line):
    return bool(_LIST_ITEM_START_PATTERN.match(stripped_line))


def _group_into_bullets(lines):
    """Groups lines so a wrapped multi-line bullet is treated as one unit."""

    chunks = []
    current = None

    for line in lines:
        stripped = line.strip()
        is_item_start = _is_list_item_start(stripped)

        if is_item_start:
            if current is not None:
                chunks.append(current)
            current = [line]
        elif stripped == "":
            if current is not None:
                chunks.append(current)
                current = None
            chunks.append([line])
        else:
            if current is not None:
                current.append(line)
            else:
                chunks.append([line])

    if current is not None:
        chunks.append(current)

    return chunks


def _filter_recommendations(text):
    """
    Bullet-level filter for Recommendations: drops any bullet/numbered
    item that references an ungrounded fix or asserts a specific
    physical direction. Logs the removal count to the console only.
    """

    chunks = _group_into_bullets(text.splitlines())
    out = []
    removed = 0

    for chunk in chunks:
        first = chunk[0].strip()
        if _is_list_item_start(first):
            joined = " ".join(l.strip() for l in chunk)
            if _is_flagged(joined):
                removed += 1
                continue
        out.extend(chunk)

    if removed:
        print(
            f"[llm_engine] _filter_recommendations: removed {removed} "
            f"recommendation(s) that referenced ungrounded fixes, "
            f"asserted an unsupported physical direction, or mislabeled "
            f"a factor's value as meters."
        )

    # Re-number whatever survived so the user never sees a gap
    # (e.g. "1. ... 3. ..." after item 2 was filtered out).
    result_lines = []
    n = 0
    for line in out:
        stripped = line.strip()
        if re.match(r"^\d+[.)]\s*", stripped):
            n += 1
            rest = re.sub(r"^\d+[.)]\s*", "", stripped)
            result_lines.append(f"{n}. {rest}")
        else:
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def build_prompt(prediction, top_features):
    """
    top_features: list of dicts, each with
        name     : human-readable feature name (e.g. "Carpet")
        value    : measured value (float)
        shap     : signed SHAP contribution (float)
        raw_name : original column name (e.g. "class_180_percent", "dx")

    This internal block is what the LLM reads to understand the
    prediction — it is never shown to the end user (see generate_report).
    """

    prompt = f"Predicted localization error: {prediction:.3f} meters\n\n"
    prompt += "Factors influencing this prediction, ranked by influence:\n"

    for i, f in enumerate(top_features, start=1):
        direction = "increases" if f["shap"] > 0 else "decreases"
        description = describe_feature(f.get("raw_name", ""))
        magnitude_label = describe_magnitude(f.get("raw_name", ""), f["value"])
        magnitude_line = f", magnitude: {magnitude_label}" if magnitude_label else ""
        prompt += (
            f"\n{i}. {f['name']} — {description}\n"
            f"   Measured value: {f['value']:.3f}{magnitude_line}\n"
            f"   Effect: {direction} the predicted error\n"
        )

    prompt += (
        "\nWrite the Explanation and Recommendations now, following the "
        "format and rules you were given."
    )

    return prompt


def _parse_sections(llm_text):
    """
    Splits the LLM's response into (explanation, recommendations) using
    the "Explanation" / "Recommendations" headers it was asked to use.
    Falls back to treating the whole response as the explanation if the
    model didn't follow the format, so the pipeline never crashes.
    """

    exp_match = re.search(
        r"Explanation\s*\n(.*?)(?:\nRecommendations\b|\Z)",
        llm_text,
        re.DOTALL | re.IGNORECASE,
    )
    rec_match = re.search(
        r"Recommendations\s*\n(.*)",
        llm_text,
        re.DOTALL | re.IGNORECASE,
    )

    explanation = exp_match.group(1).strip() if exp_match else llm_text.strip()
    recommendations = rec_match.group(1).strip() if rec_match else ""

    return explanation, recommendations


def generate_report(prediction, top_features, risk_level=None):
    """
    Produces the final, user-facing report:
      - Predicted error + risk level (written in Python, always correct)
      - Explanation: one LLM-written paragraph synthesizing the top
        factors into plain English, filtered for ungrounded claims
      - Recommendations: an LLM-written, filtered list of practical
        suggestions grounded only in the given factors

    The raw feature/value/SHAP numbers are used only to build the
    internal prompt (build_prompt) — they never appear in the output.
    """

    if risk_level is None:
        risk_level = compute_risk_level(prediction)

    prompt = build_prompt(prediction, top_features)
    llm_text = _chat(SYSTEM_PROMPT, prompt).strip()

    explanation, recommendations = _parse_sections(llm_text)
    explanation = _filter_prose(explanation, "generate_report (explanation)")
    recommendations = _filter_recommendations(recommendations)

    if not explanation:
        explanation = (
            "The predicted error for this frame comes from a combination "
            "of how much the drone moved and what the camera could see, "
            "but a clear explanation could not be generated for this run."
        )
    if not recommendations:
        recommendations = "(No grounded recommendations were generated for this frame.)"

    return (
        f"Risk Level: {risk_level}\n\n"
        f"Explanation\n{explanation}\n\n"
        f"Recommendations\n{recommendations}"
    )


# =====================================================
# SHARED OLLAMA CALL
# =====================================================

def _chat(system_prompt, user_prompt):

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:
        # An unreachable/missing Ollama server would otherwise crash the
        # whole pipeline with a raw connection traceback. Fail with a
        # clear, actionable message instead.
        raise RuntimeError(
            f"Could not reach the local Ollama server to generate the "
            f"report (model='{MODEL_NAME}'). Make sure Ollama is "
            f"installed, running, and that you've pulled the model "
            f"(`ollama pull {MODEL_NAME}`).\n"
            f"Underlying error: {exc}"
        ) from exc

    return response["message"]["content"]

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    from class_names import get_class_name

    dummy_features = [
        ("orb_features", 4400, -0.19),
        ("dx", -0.25, 0.07),
        ("class_180_percent", 1.4, 0.02),
        ("class_28_percent", 12.8, 0.05),
        ("blur", 620, 0.11),
    ]
    dummy_features.sort(key=lambda t: abs(t[2]), reverse=True)

    top_features = [
        {"name": get_class_name(f), "value": v, "shap": s, "raw_name": f}
        for f, v, s in dummy_features
    ]

    report = generate_report(prediction=0.445, top_features=top_features)

    print("\n====================================================")
    print(" UAV LOCALIZATION EXPLANATION SYSTEM (dummy demo)")
    print("====================================================\n")
    print(f"Predicted Localization Error : 0.445 meters\n")
    print(report)