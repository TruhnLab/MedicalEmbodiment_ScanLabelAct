"""The look of the task videos (pick_video.py, task_run_video.py): the console's light theme as make_video.py uses it,
fonts, chat bubbles, the list of steps, and the H.264 writer.

    from style import *                      # colours, F_* fonts, W, H, FPS
    y = bubble(draw, x, y, w, "User (chat)", text, ACCENT, USER_FILL)
    y = steps(draw, x, y, [("Find it", "done"), ("Drive up to it", "now"), ("Lift it", "todo")])
    write_video(path, frames)                # frames: RGB arrays H x W x 3
"""

import numpy as np
from PIL import ImageFont

W, H, FPS = 1920, 1080, 30

# colours (RGB): the console's light theme (console/style.css, make_video.py)
SURFACE, BORDER, TEXT, MUTED = (255, 255, 255), (225, 228, 234), (21, 24, 33), (120, 126, 138)
ACCENT, OK, BADGE, BOX = (47, 107, 255), (22, 163, 74), (241, 243, 246), (255, 196, 0)
MAP_FREE, MAP_OCC = (232, 236, 243), (21, 24, 33)
USER_FILL, ROBOT_FILL = (235, 241, 255), (236, 248, 240)


def font(bold, size):
    for f in (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else
              ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


F_TITLE, F_TEXT, F_SMALL, F_BOLD, F_BADGE = font(True, 40), font(False, 26), font(False, 21), font(True, 26), \
    font(True, 34)


def wrap(draw, text, fnt, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            test = (cur + " " + word).strip()
            if draw.textlength(test, font=fnt) <= width:
                cur = test
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def bubble(draw, x, y, w, who, text, colour, fill, max_lines=None):
    """A chat bubble; returns the y below it."""
    lines = wrap(draw, text, F_TEXT, w - 36)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;") + " …"
    h = 44 + 34 * len(lines) + 14
    draw.rounded_rectangle([x, y, x + w, y + h], 18, fill=fill, outline=BORDER)
    draw.text((x + 18, y + 10), who, font=F_SMALL, fill=colour)
    for k, line in enumerate(lines):
        draw.text((x + 18, y + 40 + 34 * k), line, font=F_TEXT, fill=TEXT)
    return y + h + 18


def steps(draw, x, y, items, title="What the robot does", width=None):
    """items: [(text, "done" | "now" | "todo")]; marks drawn (the fonts have no check mark or arrow)."""
    draw.text((x, y), title, font=F_BOLD, fill=TEXT)
    y += 46
    for text, state in items:
        cx, cy = x + 13, y + 18
        if state == "done":
            draw.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=OK)
            draw.line([(cx - 7, cy), (cx - 2, cy + 6), (cx + 7, cy - 6)], fill=SURFACE, width=4)
        elif state == "now":
            draw.polygon([(cx - 8, cy - 11), (cx - 8, cy + 11), (cx + 11, cy)], fill=ACCENT)
        else:
            draw.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], outline=MUTED, width=2)
        fnt = F_BOLD if state == "now" else F_TEXT
        for k, line in enumerate(wrap(draw, text, fnt, (width or 600) - 40)):
            draw.text((x + 40, y + 34 * k), line, font=fnt, fill=TEXT if state != "todo" else MUTED)
            y += 34 if k else 0
        y += 42
    return y


def speed_badge(draw, x, y, speed, clock_text):
    draw.rounded_rectangle([x, y, x + 120, y + 56], 14, fill=BADGE)
    draw.text((x + 22, y + 8), f"×{speed:g}", font=F_BADGE, fill=ACCENT if speed > 1 else TEXT)
    draw.text((x + 145, y + 14), clock_text, font=F_TEXT, fill=MUTED)


def write_video(path, frames):
    """frames: an iterable of RGB arrays (H x W x 3, uint8). H.264, plays in PowerPoint."""
    import imageio_ffmpeg
    writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=FPS, codec="libx264", quality=None,
                                         output_params=["-crf", "18", "-preset", "medium"], macro_block_size=1,
                                         ffmpeg_log_level="error")
    writer.send(None)
    n = 0
    for img in frames:
        writer.send(np.ascontiguousarray(img).tobytes())
        n += 1
    writer.close()
    return n
