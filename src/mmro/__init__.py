"""research_stage_00 — multimodal reasoning orientation (stage 0).

Stage 0   : load Qwen2.5-VL-7B and InternVL3-8B, prove multi-image + high-res
            inputs work end to end, and measure the OOM boundary on a 24GB L4.
Stage 0.1 : vLLM serving path (isolated venv).
Stage 0.2 : MMMU / MathVista via VLMEvalKit *and* lmms-eval, scores compared.
"""

__version__ = "0.1.0"
