"""Clef-format calibration and held-out eval records for the Clef-Omni quants (issue #152).

Byte-for-byte the generator used by the ModelOpt NVFP4 / NVFP4A16 jobs: text from ultrachat_200k
test_sft wrapped in Clef's state/questions schema (choice / noul / score), ~20 % synthetic image /
video / audio. Same seeds and split, so every quant is graded on the identical 128 held-out records.
"""
import math
import random
import time

import numpy as np


def log(*a):
    print(f"[{time.strftime('%FT%T')}]", *a, flush=True)


def synth_image(rng):
    from PIL import Image, ImageDraw
    w, h = int(rng.integers(224, 641)), int(rng.integers(224, 641))
    im = Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(3, 12))):
        x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        box = (x0, y0, x0 + int(rng.integers(10, w // 2)), y0 + int(rng.integers(10, h // 2)))
        col = tuple(int(x) for x in rng.integers(0, 255, 3))
        (d.ellipse if rng.random() < .5 else d.rectangle)(box, fill=col)
    return im

def synth_video(rng):
    n, s = int(rng.choice([4, 6, 8])), int(rng.choice([128, 160, 224]))
    x, y = np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s))
    k = rng.uniform(2, 12, 3)
    return np.stack([(np.stack([0.5 + 0.5 * np.sin(x * k[0] + i), y ** (1 + i / n),
                                0.5 + 0.5 * np.cos(y * k[2] - i * k[1] / 6)], -1) * 255).astype("uint8")
                     for i in range(n)])

def synth_audio(rng, rate=16000):
    t = np.arange(int(rng.uniform(1.0, 4.0) * rate), dtype="float32") / rate
    sig = sum(rng.uniform(.05, .3) * np.sin(2 * math.pi * rng.uniform(80, 2000) * t) for _ in range(3))
    sig = sig + rng.uniform(0, .05) * rng.standard_normal(len(t))
    return (sig * np.hanning(len(t))).astype("float32")

def load_texts():
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        p = hf_hub_download("HuggingFaceH4/ultrachat_200k", repo_type="dataset",
                            filename="data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet")
        rows = pq.read_table(p, columns=["prompt", "messages"]).to_pylist()
        out = []
        for r in rows:
            reply = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
            out.append((r["prompt"][:1200], reply[:2400]))
        log(f"ultrachat test_sft rows: {len(out)}")
        return out
    except Exception as e:
        log("ultrachat unavailable, synthetic text only:", repr(e))
        words = "agent player wall door enemy ladder coin map key light shadow path river bridge tower".split()
        rng = random.Random(7)
        return [(" ".join(rng.choices(words, k=30)), " ".join(rng.choices(words, k=120))) for _ in range(4000)]

VERBS = ["move forward", "turn left", "turn right", "wait", "jump", "retreat", "interact", "stop"]
def make_record(i, text, rng):
    prompt, reply = text
    state = {"task": prompt, "observation": reply}
    if rng.random() < .3:
        state["history"] = "\n".join(f"step {j}: {rng.choice(VERBS)}" for j in range(int(rng.integers(2, 40))))
    qs = {}
    for qi in range(int(rng.integers(1, 4))):
        kind = rng.choice(["choice", "noul", "score"])
        if kind == "choice":
            opts = list(rng.choice(VERBS, size=int(rng.integers(2, 6)), replace=False))
            qs[f"q{qi}"] = {"type": "choice", "instructions": "Which response best serves the task?",
                            "criteria": {o.replace(" ", "_"): o for o in opts}}
        elif kind == "noul":
            qs[f"q{qi}"] = {"type": "noul", "instructions": "Does the observation answer the task correctly?"}
        else:
            qs[f"q{qi}"] = {"type": "score", "instructions": "Rate the observation's helpfulness",
                            "criteria": ["useless", "weak", "adequate", "good", "excellent"][: int(rng.integers(3, 6))]}
    rec = {"id": f"r{i}", "state": state, "questions": qs}
    u = rng.random()
    if u < .10:
        rec["images"] = [synth_image(rng)]
    elif u < .15:
        rec["videos"] = [synth_video(rng)]
    elif u < .20:
        rec["audio"] = [synth_audio(rng)]
    return rec



def build(n_calib=512, n_eval=128):
    texts = load_texts()
    order = list(range(len(texts))); random.Random(145).shuffle(order)
    calib_idx, eval_idx = order[:n_calib], order[n_calib:n_calib + n_eval]
    rng_c, rng_e = np.random.default_rng(1), np.random.default_rng(2)
    calib = [make_record(i, texts[i], rng_c) for i in calib_idx]
    evals = [make_record(i, texts[i], rng_e) for i in eval_idx]
    return calib, evals
