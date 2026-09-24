"""Launch VLMEvalKit's run.py with extra kwargs for its own model class.

Runs INSIDE the VLMEvalKit venv (by path, not as part of mmro), so it may import
only the standard library and vlmeval.

Why this exists: ``run.py --model X --data Y`` has no way to pass constructor
kwargs to X. Its ``--use-vllm`` flag looks like one, but it ALSO sets
``judge_kwargs['use_vllm']``, which ``build_judge`` hands to ``OpenAIWrapper``
as a default request kwarg -- every judge call then carries ``"use_vllm": true``
in its JSON body and the OpenAI API rejects it. The judge fails quietly and the
score drops. ``--config`` is the documented way to pass model kwargs without
touching the judge, so this writes that JSON and hands over to run.py.

The model entry names the harness's registered model (so its registered
defaults still apply) plus our kwargs. The data entry is keyed by the dataset
name itself, so output filenames match a plain ``--data`` run and the score
parser is unaffected. Its class is resolved exactly as ``build_dataset`` does.

Usage:
    python vlmevalkit_launch.py RUN_PY --model M --data D --model-kwargs JSON
        --config-out PATH [run.py args...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def dataset_class(name: str) -> str:
    import vlmeval.dataset as vd

    for cls in vd.DATASET_CLASSES:
        if name in cls.supported_datasets():
            if not hasattr(vd, cls.__name__):
                sys.exit(f"vlmevalkit_launch: {cls.__name__} (for {name}) is not "
                         f"exported by vlmeval.dataset, so --config cannot name it")
            return cls.__name__
    sys.exit(f"vlmevalkit_launch: no VLMEvalKit dataset class supports {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-kwargs", default="{}")
    parser.add_argument("--config-out", required=True)
    args, passthrough = parser.parse_known_args()

    model_kwargs = json.loads(args.model_kwargs)
    if not model_kwargs:
        # Nothing to add: the plain CLI path, byte-for-byte what it was before.
        argv = ["--model", args.model, "--data", args.data]
    else:
        from vlmeval.config import supported_VLM

        if args.model not in supported_VLM:
            sys.exit(f"vlmevalkit_launch: {args.model!r} is not a registered "
                     f"VLMEvalKit model")
        config = {
            # No "class" key: run.py then calls supported_VLM[name](**kwargs),
            # i.e. the registered partial with our kwargs layered on top.
            "model": {args.model: model_kwargs},
            "data": {args.data: {"class": dataset_class(args.data), "dataset": args.data}},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.config_out)), exist_ok=True)
        with open(args.config_out, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        print(f"vlmevalkit_launch: wrote {args.config_out}: {json.dumps(config)}", flush=True)
        argv = ["--config", args.config_out]

    if model_kwargs.get("use_vllm"):
        # vLLM's own recommendation, which VLMEvalKit warns about when unset.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    os.execv(sys.executable, [sys.executable, args.run_py, *argv, *passthrough])


if __name__ == "__main__":
    # Running by path puts this directory first on sys.path; its sibling
    # modules (base.py, lmms_eval.py) must not shadow anything vlmeval imports.
    sys.path.pop(0)
    main()
