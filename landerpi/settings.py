"""Site-specific settings (model gateway, GPU cluster): from the environment, else from the repo's .env (not in
git; .env.example lists the names). They stay out of the code because the repository is public (2026-10-14).

    from landerpi.settings import setting
    url = setting("VLLM_URL")                          # raises if it is not set anywhere
    master = setting("CLUSTER_MASTER", "user@slurm-master")   # placeholder default
"""

import os
import re
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def setting(name, default=None):
    if os.environ.get(name):
        return os.environ[name]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(rf"\s*{name}=(\S+)", line)
            if m:
                return m[1]
    if default is None:
        raise RuntimeError(f"{name} is not set (environment or {ENV_FILE})")
    return default
