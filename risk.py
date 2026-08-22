"""Rule-based osteoarthritis risk display.

Moved out of the old `vision.py` when the simulated X-ray module was removed.
Nothing here is a trained model -- these are transparent thresholds taken from
well-established OA risk factors, computed from what the clinician entered.
"""


def risk_factors(age, bmi, previous_injury, family_history, physical_load):
    """Return a list of {name, level, value} rows for the risk panel."""

    def band(value, moderate, high):
        if value >= high:
            return "High", 0.9
        if value >= moderate:
            return "Moderate", 0.6
        return "Low", 0.3

    age_label, age_val = band(age, 50, 65)
    bmi_label, bmi_val = band(bmi, 25, 30)

    def yes_no(flag, high_label="Moderate", high_val=0.6):
        return (high_label, high_val) if flag else ("Low", 0.3)

    injury_label, injury_val = yes_no(previous_injury)
    family_label, family_val = yes_no(family_history)
    load_label, load_val = yes_no(physical_load)

    return [
        {"name": "Age (> 50)", "level": age_label, "value": age_val},
        {"name": "High BMI", "level": bmi_label, "value": bmi_val},
        {"name": "Previous Injury", "level": injury_label, "value": injury_val},
        {"name": "Family History", "level": family_label, "value": family_val},
        {"name": "High Physical Load", "level": load_label, "value": load_val},
    ]


def overall_risk(rows):
    """Coarse overall band from the individual factor rows."""
    if not rows:
        return "Low", 0.0
    score = sum(row["value"] for row in rows) / len(rows)
    if score >= 0.7:
        return "High", score
    if score >= 0.45:
        return "Moderate", score
    return "Low", score
