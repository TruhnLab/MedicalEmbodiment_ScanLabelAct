"""The consoles' JavaScript libraries (console/vendor/): downloaded on the helper's first start instead of kept in
git, at pinned versions with checked hashes. The public repository carries no third-party code (user,
2026-10-14); the files were byte-identical to these package files when they left git.

    python tools/console_vendor.py          # check / download without the helper

A file that is missing or differs is downloaded again; a download whose hash doesn't match is an error and is
not written.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1] / "console" / "vendor"

# path in console/vendor/, source, sha256. Used by index.html, tasks.html and tasks_boot.js (roslib, pako,
# UPNG) and by the 3D views through their import maps (three + two addons).
FILES = [
    ("roslib.min.js", "https://unpkg.com/roslib@1.4.1/build/roslib.min.js",
     "3c510df2bc04be9d5b07efafb01eb5f473d8a5092d93292351a715ca60daf9d6"),
    ("pako.min.js", "https://unpkg.com/pako@2.1.0/dist/pako.min.js",
     "ede2693a4a6a5126b9d35669062b358ecab6ae7b9b86a1cf302feb45a8514907"),
    ("UPNG.js", "https://unpkg.com/upng-js@2.1.0/UPNG.js",
     "b7c0bdb021dffeb82f1ac27c6762f939f967a9e4e0886518fef649331b612164"),
    ("three/three.core.min.js", "https://unpkg.com/three@0.180.0/build/three.core.min.js",
     "61ba0df005b05991361d040d8ff670e1aadfd0ce7aeebd1fdb0725957a8957de"),
    ("three/three.module.min.js", "https://unpkg.com/three@0.180.0/build/three.module.min.js",
     "e2b5ee6bccd38fd6d8a2428546b83c5f2426d84b152ef82be8055556e3b40eb6"),
    ("three/addons/controls/OrbitControls.js",
     "https://unpkg.com/three@0.180.0/examples/jsm/controls/OrbitControls.js",
     "b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c"),
    ("three/addons/loaders/PLYLoader.js", "https://unpkg.com/three@0.180.0/examples/jsm/loaders/PLYLoader.js",
     "b824a6dda90cbfb37cabf8e5462ed1673e7693852bc1a6a7bab0465d7b634dfb"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(root=VENDOR_DIR, timeout=20, log=print):
    """Makes every file of FILES present and correct under root. Returns the paths still missing (empty: all
    fine). Stops downloading after the first network error (no internet: don't wait for every file)."""
    missing, offline = [], False
    for rel, url, digest in FILES:
        target = root / rel
        if target.is_file() and sha256(target) == digest:
            continue
        if offline:
            missing.append(rel)
            continue
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
        except OSError as e:
            log(f"console libraries: could not download {url}: {e}")
            missing.append(rel)
            offline = True
            continue
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            log(f"console libraries: {url} has sha256 {got}, expected {digest}: not used")
            missing.append(rel)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        log(f"console libraries: downloaded {rel} ({len(data) // 1024} KB)")
    return missing


if __name__ == "__main__":
    left = ensure()
    print("console libraries: all present" if not left else f"console libraries missing: {', '.join(left)}")
    sys.exit(1 if left else 0)
