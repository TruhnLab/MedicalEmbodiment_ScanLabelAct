"""Videos and pictures for the page, from the code repository's presentation material.

    conda run -n landerpi python tools/make_media.py --frames <folder of tools/capture_orbit.mjs>

Sources (../LanderPi/data/): the kitchen's exploration video (projects/presentation_video/make_video.py), the
find and pick videos and the localization figure (data/demo/); and the frames of the labelled room turning,
captured from the page's own 3D viewer by tools/capture_orbit.mjs (left out of the teaser without --frames).
Writes assets/media/: explore.mp4, find.mp4, pick.mp4, teaser.mp4 (1280 x 720, H.264, no sound) with a poster
JPG each, localization.png and labels.jpg.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

PAGE = Path(__file__).resolve().parents[1]
DATA = PAGE.parent / "LanderPi" / "data"
OUT = PAGE / "assets" / "media"
FF = imageio_ffmpeg.get_ffmpeg_exe()

EXPLORE = DATA / "sessions" / "20261007_101108" / "video" / "20261007_101108_x32.mp4"
FIND = DATA / "demo" / "20261009_090645_x2.mp4"
PICK = DATA / "demo" / "pick_163719.mp4"

# the slides' text has to stay readable at 720p; CRF 27 keeps them sharp at ~100 kB/s
ENCODE = ["-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"]
SCALE = "scale=1280:720:flags=lanczos"

# teaser: (source, start s, end s, speed-up); the find clip starts after the first seconds (a person walks by)
def teaser_parts(orbit):
    return [(EXPLORE, 0.0, 32.0, 4.0), (orbit, 0.0, None, 1.0), (FIND, 3.0, 21.3, 2.0), (PICK, 0.0, 29.5, 4.0)]


def ff(*args):
    subprocess.run([FF, "-y", "-loglevel", "error", *map(str, args)], check=True)


def poster(video, out, at):
    ff("-ss", at, "-i", video, "-frames:v", "1", "-vf", SCALE, "-q:v", "3", out)


def duration(video):
    cap = cv2.VideoCapture(str(video))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, help="folder with the orbit frames (0000.png ...) and labels.png")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    orbit = Path("missing")
    if args.frames:
        orbit = args.frames / "orbit.mp4"
        ff("-framerate", 30, "-i", args.frames / "%04d.png", *ENCODE, orbit)
        ff("-i", args.frames / "labels.png", "-q:v", "3", OUT / "labels.jpg")

    for name, src, at in [("explore", EXPLORE, 31.5), ("find", FIND, 21.0), ("pick", PICK, 29.0)]:
        ff("-i", src, "-vf", f"{SCALE},fps=30", *ENCODE, OUT / f"{name}.mp4")
        poster(src, OUT / f"{name}.jpg", at)
        print(f"{name}.mp4 {(OUT / f'{name}.mp4').stat().st_size / 1e6:.1f} MB")

    # localization figure: the three panels without the slide's empty margins
    img = cv2.imread(str(DATA / "demo" / "localization.png"))
    cv2.imwrite(str(OUT / "localization.png"), img[176:880, 24:1896], [cv2.IMWRITE_PNG_COMPRESSION, 9])

    parts, inputs, starts, t = [], [], [], 0.0
    for src, a, b, speed in teaser_parts(orbit):
        if not src.exists():
            print(f"teaser: {src.name} missing, left out")
            continue
        b = duration(src) if b is None else b
        inputs += ["-i", src]
        n = len(inputs) // 2 - 1
        parts.append(f"[{n}:v]trim={a}:{b},setpts=(PTS-STARTPTS)/{speed},{SCALE},fps=30,setsar=1[v{n}]")
        starts.append(round(t, 2))
        t += (b - a) / speed
    concat = "".join(f"[v{n}]" for n in range(len(parts)))
    ff(*inputs, "-filter_complex", ";".join(parts) + f";{concat}concat=n={len(parts)}:v=1[out]", "-map", "[out]",
       *ENCODE, OUT / "teaser.mp4")
    poster(OUT / "teaser.mp4", OUT / "teaser.jpg", 13.0)  # the labelled room
    print(f"teaser.mp4 {(OUT / 'teaser.mp4').stat().st_size / 1e6:.1f} MB, {t:.1f} s, parts start at {starts}")


if __name__ == "__main__":
    main()
