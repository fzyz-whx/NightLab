"""NightLab five-stage normalization; evidence: docs/LABEL_MAPPING_EVIDENCE.md."""

STAGES = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N3', 5: 'REM'}


def normalize_stage(raw_stage):
    """Retain source code and an explicit target mask; reject undocumented codes."""
    if type(raw_stage) is not int or raw_stage not in {-1, *STAGES}:
        raise ValueError(f'undocumented SleepAccel label: {raw_stage!r}')
    return {'raw_stage': raw_stage, 'stage': STAGES.get(raw_stage),
            'target_valid': raw_stage != -1}
