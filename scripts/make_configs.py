"""Regenerate configs/*.json from code defaults (run after changing config fields)."""
import copy
import json
from pathlib import Path

from sfu.config import ExperimentConfig, from_dict

base = ExperimentConfig().to_dict()

demo = copy.deepcopy(base)
demo["design"].update(trials_per_block=36, practice_trials=6, calibration_block=True, calibration_trials_per_level=6)
demo["output_dir"] = "data/demo"
demo["keyboard_backend"] = "event"  # works on macOS without Input Monitoring permission

probe = copy.deepcopy(demo)
probe["design"]["paradigm"] = "probe"
probe["output_dir"] = "data/demo_probe"

pilot = copy.deepcopy(base)
pilot["monitor"].update(name="REPLACE-WITH-MEASURED-VALUES", calibrated=False)
pilot["design"]["calibration_block"] = True
pilot["output_dir"] = "data/pilot"

for name, d in {"demo": demo, "demo_probe": probe, "pilot_template": pilot}.items():
    from_dict(copy.deepcopy(d))  # validate
    Path(f"configs/{name}.json").write_text(json.dumps(d, indent=2))
    print("wrote", name)
