"""A second opinion on candidates: before a sighting counts as found, a vision-language model on the vLLM
gateway looks at its crop and judges whether it fits the user's description (user, 2026-10-09: three cubes, red,
blue and green, were all taken for "small green block": YOLOE weighs the shape and hardly the colour, and its
label is only the prompt). Not the chat: no conversation, only the description and the crops; its own thread,
so the robot keeps driving and the detector keeps running while it judges.

    v = Verifier(on_verdict)                   # on_verdict(entry, fits: True/False/None, what: str)
    v.submit([entry, ...], "small green cube, like a dice")

Only a few calls per search (live.py / perception.Watcher decide when): a sighting once its box was followed in
two images; again when its box has grown to twice the size of its last check (at most 3). Candidates that
come together go into one request as numbered crops (comparing helps: "the green one is 2"). If the gateway
does not answer, fits is None: the candidate counts as before ("not checked"), the robot does not depend on it.
The chat's key (VLLM_CHAT_API_KEY): this is part of the chat feature; the labelling keeps its own.
"""

import base64
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_understanding"))
import vlm  # noqa: E402  the gateway's settings; stopgap import, see README

BATCH_WAIT_S = 0.4     # candidates arriving this close together go into one request
MAX_BATCH = 4
MIN_CROP_PX = 224      # small crops are enlarged (a cube 2 m away is ~25 px)
TIMEOUT_S = 20

QUESTION = """A robot is looking for: "{description}".
Below are {n} numbered camera crops; each shows a candidate in its middle, marked by small grey corner marks \
(other things around it, e.g. in a pile, are not the candidate). For each crop, does the candidate fit the \
description? Judge what the candidate is and how it looks (kind of object, colour, size relative to its \
surroundings, shape); ignore any words about where it is (next to what, between what): a crop cannot show that. \
Decide for every crop: true or false (a small or blurry one by what you can see; it is looked at again closer).
Answer as JSON only: {{"crops": [{{"n": 1, "fits": true or false, "what": "what it is, a few words"}}, ...]}}"""


class Verifier:
    def __init__(self, on_verdict):
        self.on_verdict = on_verdict
        self.q = queue.Queue()
        self.calls = 0
        self.last_ms = None
        assert not vlm.blocked(vlm.MODEL)
        threading.Thread(target=self.run, daemon=True, name="verifier").start()

    def submit(self, entries, description):
        for e in entries:
            self.q.put((e, description))

    def run(self):
        while True:
            batch = [self.q.get()]
            time.sleep(BATCH_WAIT_S)
            while len(batch) < 4 * MAX_BATCH:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            groups = {}
            for e, desc in batch:
                groups.setdefault(desc, []).append(e)
            for desc, entries in groups.items():
                for i in range(0, len(entries), MAX_BATCH):
                    try:
                        self.judge(desc, entries[i:i + MAX_BATCH])
                    except Exception as e:  # never end the thread: a dead verifier left every candidate
                        # "checking" for good (real run 2026-10-09, a KeyError on "unsure")
                        print(f"verifier: {type(e).__name__}: {e}", flush=True)

    def judge(self, description, entries):
        unread = None
        try:
            answers = self.ask(description, [e["crop"] for e in entries])
            failed = "not checked (no answer for this crop)"
        except requests.RequestException as e:  # no gateway: the candidates count unchecked (as before the check)
            answers, failed = {}, f"not checked ({type(e).__name__})"
        except Exception as e:  # an answer it could not read: not a find yet, checked again when seen larger
            answers, unread = {}, f"answer not readable ({type(e).__name__})"
            print(f"verifier: {unread}: {e}", flush=True)
        for n, entry in enumerate(entries, 1):
            a = answers.get(n)
            if a is None and unread:
                fits, what = False, unread  # no find for now; looked at again when seen twice as large
            elif a is None:
                fits, what = None, failed
            else:
                f = a.get("fits")
                # no "unsure" any more (user, 2026-10-12: it caused trouble; every result is looked at again when
                # the box has grown to twice the size): anything but true is "does not fit" for now
                fits = f.lower() == "true" if isinstance(f, str) else bool(f)
                what = str(a.get("what") or "")[:80]
            try:  # each candidate on its own: one failing must not leave the others "checking"
                self.on_verdict(entry, fits, what)
            except Exception as e:
                print(f"verifier: {type(e).__name__}: {e}", flush=True)

    def ask(self, description, crops):
        from assistant import chat_key  # the chat's key (not the labelling's)
        content = [{"type": "text", "text": QUESTION.format(description=description, n=len(crops))}]
        for n, jpeg in enumerate(crops, 1):
            content += [{"type": "text", "text": f"Crop {n}:"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + enlarged(jpeg)}}]
        body = {"model": vlm.MODEL, "temperature": 0, "max_tokens": 80 + 60 * len(crops),
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": content}]}
        t0 = time.time()
        r = requests.post(vlm.URL, json=body, headers={"Authorization": "Bearer " + chat_key()}, verify=False,
                          timeout=TIMEOUT_S)
        r.raise_for_status()
        self.calls += 1
        self.last_ms = round((time.time() - t0) * 1000)
        answer = vlm.parse_json(r.json()["choices"][0]["message"]["content"] or "{}")
        # {"crops": [...]}, or just the list, or for one crop just its object (real run 2026-10-09: a bare list
        # raised an AttributeError, the candidates counted as "not checked" and a red block was "found" where
        # the blue one lay)
        items = answer.get("crops", [answer] if "fits" in answer else []) if isinstance(answer, dict) else answer
        out = {}
        for i, a in enumerate(items if isinstance(items, list) else [], 1):
            if isinstance(a, dict):
                out[int(a.get("n", i))] = a
        return out


def enlarged(jpeg):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    k = MIN_CROP_PX / max(1, min(img.shape[:2]))
    if k > 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()
