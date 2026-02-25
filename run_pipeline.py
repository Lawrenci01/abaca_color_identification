"""
RUN_PIPELINE.py — Master runner for the Abaca Color ML Pipeline (Upgraded)
===========================================================================
Executes all steps in sequence:
  Step 1 → Extract RHS color data         (extract_colors.py)
  Step 2 → Generate augmented dataset     (augment_dataset.py)  150 aug/color
  Step 3 → Train RF + SVM ensemble        (train_model.py)
  Step 4 → Evaluate and produce report    (evaluate.py)
  Step 5 → Launch inference server        (inference_server.py)

Usage:
  python run_pipeline.py          # runs Steps 2-4 (skips Step 1 if CSV exists)
  python run_pipeline.py --all    # runs Steps 1-4 (re-extracts colors)
  python run_pipeline.py --train  # runs Steps 2-3 only
  python run_pipeline.py --eval   # runs Step 4 only (models must exist)
  python run_pipeline.py --serve  # runs Step 5 only (models must exist)
"""

import sys
import importlib.util
from pathlib import Path


def run_step(script_path, description):
    print(f"\n{'='*65}")
    print(f"  {description}")
    print(f"{'='*65}")
    spec   = importlib.util.spec_from_file_location("module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    args      = sys.argv[1:]
    base_dir  = Path(__file__).parent

    steps = {
        'step1': (base_dir / "extract_colors.py",    "STEP 1: Extract RHS colors from PDF"),
        'step2': (base_dir / "augment_dataset.py",   "STEP 2: Generate augmented dataset (150 aug/color)"),
        'step3': (base_dir / "train_model.py",        "STEP 3: Train RF + SVM ensemble model"),
        'step4': (base_dir / "evaluate.py",           "STEP 4: Evaluate model & generate report"),
        'step5': (base_dir / "inference_server.py",   "STEP 5: Launch iPhone inference server"),
    }

    if '--serve' in args:
        run_step(*steps['step5'])

    elif '--eval' in args:
        run_step(*steps['step4'])

    elif '--train' in args:
        for k in ['step2', 'step3']:
            run_step(*steps[k])

    elif '--all' in args:
        for k in ['step1', 'step2', 'step3', 'step4']:
            run_step(*steps[k])

    else:
        # Default: skip step1 if rhs_colors.csv already exists
        csv_path = Path("abaca_pipeline/rhs_colors.csv")
        if not csv_path.exists():
            print("rhs_colors.csv not found — running Step 1 first ...")
            run_step(*steps['step1'])
        else:
            print(f"✅  Found existing rhs_colors.csv ({sum(1 for _ in open(csv_path))-1} colors)")

        for k in ['step2', 'step3', 'step4']:
            run_step(*steps[k])

        print(f"\n{'='*65}")
        print(f"  ✅  Pipeline complete! Ready for iPhone testing.")
        print(f"  Run:  python run_pipeline.py --serve")
        print(f"{'='*65}")