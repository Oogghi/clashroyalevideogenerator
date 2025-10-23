import os
import random
import math
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import subprocess

# ---------- CONFIG ----------
OUT_DIR = "output_videos"
BACKGROUND_DIR = "Menu"
CARDS_DIR = "Cards"
ITEMS_DIR = "Items"
FONT_PATH = "Fonts/Clash_Regular.otf"  # provided font
VIDEO_SIZE = (1080, 1920)  # vertical 9:16 (W, H)
FPS = 60
# Phases
INTRO_DUR = 2.5
GUESS_DUR = 4
REVEAL_DUR = 3
OUTRO_SUBSCRIBE_DUR = 5
DURATION_PER_CARD = INTRO_DUR + GUESS_DUR + REVEAL_DUR
COUNT_BY_RARITY = {
    "Common": 2,
    "Rare": 2,
    "Epic": 1,
    "Legendary": 1,
    "Champion": 1
}
TITLE_TEXT = "Guess the card!"
TITLE_FONT_SIZE = 86
SUBTEXT_FONT_SIZE = 48
NAME_FONT_SIZE = 64
CONCEPT_FONT_SIZE = 72  # large centered concept text during intro
TEXT_COLOR = (255, 255, 255, 255)  # RGBA
DIFFICULTY_BY_RARITY = {
    "Common": "Easy",
    "Rare": "Medium",
    "Epic": "Hard",
    "Legendary": "Expert",
    "Champion": "Impossible"
}

# ---------- NEW: GLIMPSE CONFIG ----------
# How long to briefly show the full (non-pixelated) card during the GUESS phase, per rarity.
# Values are in seconds. Set Champion to 1.0/FPS (one frame) if you want exactly one frame.
# You can change these values per-run.
GLIMPSE_DURATION_BY_RARITY = {
    "Common": 8 / FPS,
    "Rare": 6.0 / FPS,
    "Epic": 4.0 / FPS,
    "Legendary": 2.0 / FPS,
    "Champion": 1.01 / FPS
}
# Where inside the GUESS phase the glimpse will appear (0.0=start .. 1.0=end). 0.5 centers it.
GLIMPSE_POSITION_IN_GUESS = 0.20
# Opacity for the glimpse (0-1). 1.0 = fully visible.
GLIMPSE_OPACITY_BY_RARITY = {
    "Common": 1.0,
    "Rare": 1.0,
    "Epic": 1.0,
    "Legendary": 1.0,
    "Champion": 1.0
}

# ---------- UTIL ----------
def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def find_item_by_contains(items, token):
    token = token.lower()
    for p in items:
        if token in os.path.basename(p).lower():
            return p
    return None

def fit_image_preserve(img_path, max_size):
    """Load image and resize to fit within max_size preserving aspect ratio (no stretch)."""
    im = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = im.size
    max_w, max_h = max_size
    if orig_w == 0 or orig_h == 0:
        return im.resize((max_w, max_h), Image.LANCZOS)
    ratio = min(max_w / orig_w, max_h / orig_h)
    target_w = max(1, int(orig_w * ratio))
    target_h = max(1, int(orig_h * ratio))
    return im.resize((target_w, target_h), Image.LANCZOS)

def pil_text_image(text, font_path, fontsize, color=(255,255,255,255), max_width=900, align="center", padding=20):
    try:
        font = ImageFont.truetype(font_path, fontsize)
    except Exception as e:
        print("Warning: font load failed:", e, "- using default font")
        font = ImageFont.load_default()
    temp_img = Image.new("RGBA", (10, 10), (0,0,0,0))
    temp_draw = ImageDraw.Draw(temp_img)
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = temp_draw.textbbox((0,0), test, font=font)
        test_w = bbox[2] - bbox[0]
        if test_w <= max_width - 2*padding:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    line_sizes = []
    total_h = 0
    max_w = 0
    for line in lines:
        bbox = temp_draw.textbbox((0,0), line, font=font)
        w_line = bbox[2] - bbox[0]
        h_line = bbox[3] - bbox[1]
        line_sizes.append((w_line, h_line))
        total_h += h_line
        if w_line > max_w:
            max_w = w_line
    inter = 6
    text_h = total_h + max(0, (len(lines)-1))*inter
    img_w = max_w + 2*padding
    img_h = text_h + 2*padding
    img = Image.new("RGBA", (img_w, img_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    y = padding
    for idx, line in enumerate(lines):
        w_line, h_line = line_sizes[idx]
        if align == "center":
            x = (img_w - w_line)//2
        elif align == "right":
            x = img_w - padding - w_line
        else:
            x = padding
        draw.text((x, y), line, font=font, fill=color)
        y += h_line + inter
    return img

def pil_to_bgr(pil_img):
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    bgr = arr[:, :, ::-1].copy()
    return bgr

def apply_alpha(pil_img, alpha):
    # alpha in [0,1]
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    a = pil_img.split()[-1]
    a = a.point(lambda p: int(p * alpha))
    pil_img.putalpha(a)
    return pil_img

def ease_out_cubic(x):
    return 1 - pow(1 - x, 3)

# write one card video
def write_card_video(card_path, background_path, items_dir, rarity, outpath,
                     duration=DURATION_PER_CARD, fps=FPS, video_size=VIDEO_SIZE,
                     include_intro=True, outro_subscribe=False):
    """
    Instead of pixelating, this version briefly *shows* the full card during the GUESS phase
    according to GLIMPSE_DURATION_BY_RARITY. For 'Champion' (Impossible) the default is 1 frame.
    """
    W, H = video_size
    # load items list
    items_all = list_images(items_dir)
    # find cube icon: Item_Lucky_Drop_{rarity}
    rarity_token = rarity.lower()
    cube_path = None
    candidates = [f"item_lucky_drop_{rarity_token}", f"item_lucky_drop_{rarity_token}_", rarity_token + "_card", rarity_token]
    for c in candidates:
        p = find_item_by_contains(items_all, c)
        if p:
            cube_path = p
            break
    # secret card image (e.g., common_card.png)
    secret_card_path = find_item_by_contains(items_all, rarity_token + "_card")
    # Background
    bg = Image.open(background_path).convert("RGBA").resize((W, H), Image.LANCZOS)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 90))  # semi-transparent overlay

    # compute card bounding box (no stretching)
    max_card_h = int(H * 0.45)
    max_card_w = int(max_card_h * 0.65)

    # full (revealed) card image, aspect-preserving
    full_card_img = fit_image_preserve(card_path, (max_card_w, max_card_h))
    # the glimpse image is just the same (non-pixelated). We may apply different opacity.
    glimpse_img = full_card_img.copy()

    # prepare cube
    cube_img = None
    if cube_path:
        try:
            c = Image.open(cube_path).convert("RGBA")
            max_cube_w = int(W * 0.6)
            cw, ch = c.size
            new_w = min(max_cube_w, cw)
            new_h = int(ch * (new_w / cw))
            cube_img = c.resize((new_w, new_h), Image.LANCZOS)
        except Exception as e:
            print("Warning cube load:", e)
            cube_img = None
    # secret card
    secret_card_img = None
    if secret_card_path:
        try:
            s = Image.open(secret_card_path).convert("RGBA")
            sw, sh = s.size
            sc_target_w = int(max_card_w * 0.95)
            ratio = min(sc_target_w / sw, (max_card_h // 2) / sh)
            if ratio <= 0:
                ratio = min(sc_target_w / sw, 1.0)
            new_w = max(1, int(sw * ratio))
            new_h = max(1, int(sh * ratio))
            secret_card_img = s.resize((new_w, new_h), Image.LANCZOS)
        except Exception as e:
            print("Warning secret card load:", e)
            secret_card_img = None

    # --- Prepare thumbs (for outro_subscribe) ---
    THUMBS_MAX_WIDTH_RATIO = 0.85
    THUMBS_ALLOW_UPSCALE = True
    THUMBS_MAX_UPSCALE = 3.0

    thumbs_img = None
    if outro_subscribe:
        thumbs_path = find_item_by_contains(items_all, "thumbs")
        if not thumbs_path:
            fallback = os.path.join(items_dir, "Thumbs.png")
            if os.path.exists(fallback):
                thumbs_path = fallback
        if thumbs_path:
            try:
                timg = Image.open(thumbs_path).convert("RGBA")
                tmax_w = int(W * THUMBS_MAX_WIDTH_RATIO)
                tw, th = timg.size
                if tw <= 0:
                    scale = 1.0
                else:
                    desired_scale = float(tmax_w) / float(tw)
                    if not THUMBS_ALLOW_UPSCALE:
                        scale = min(1.0, desired_scale)
                    else:
                        scale = max(0.001, desired_scale)
                        if THUMBS_MAX_UPSCALE is not None:
                            scale = min(scale, THUMBS_MAX_UPSCALE)
                new_tw = max(1, int(round(tw * scale)))
                new_th = max(1, int(round(th * scale)))
                thumbs_img = timg.resize((new_tw, new_th), Image.LANCZOS)
            except Exception as e:
                print("Warning thumbs load:", e)
                thumbs_img = None

    # prepare text images
    difficulty = DIFFICULTY_BY_RARITY.get(rarity, "Easy")
    title_img = pil_text_image(TITLE_TEXT, FONT_PATH, TITLE_FONT_SIZE, color=TEXT_COLOR, max_width=int(W*0.9))
    subtext = f"{difficulty}"
    sub_img = pil_text_image(subtext, FONT_PATH, SUBTEXT_FONT_SIZE, color=TEXT_COLOR, max_width=int(W*0.9))
    concept_text_intro = f"Can you guess the CARD? It will flash faster and faster..."
    concept_img_intro = pil_text_image(concept_text_intro, FONT_PATH, CONCEPT_FONT_SIZE, color=TEXT_COLOR, max_width=int(W*0.95))
    subscribe_text = "Subscribe for more Clash Royale tests! Give your ANSWER in the comments!"
    subscribe_img = pil_text_image(subscribe_text, FONT_PATH, CONCEPT_FONT_SIZE - 6, color=TEXT_COLOR, max_width=int(W*0.95), align="center")
    base_name = os.path.splitext(os.path.basename(card_path))[0]
    name_img = pil_text_image(base_name, FONT_PATH, NAME_FONT_SIZE, color=TEXT_COLOR, max_width=int(W*0.9))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(outpath, fourcc, fps, (W, H))
    bob_amp = 8
    bob_period = 2.5

    intro_dur = INTRO_DUR if include_intro else 0.0
    if outro_subscribe:
        reveal_dur = OUTRO_SUBSCRIBE_DUR
    else:
        reveal_dur = REVEAL_DUR
    duration = intro_dur + GUESS_DUR + reveal_dur
    total_frames = int(math.ceil(duration * fps))

    # compute glimpse timing parameters (in seconds)
    glimpse_dur = GLIMPSE_DURATION_BY_RARITY.get(rarity, 0.25)
    glimpse_opacity = GLIMPSE_OPACITY_BY_RARITY.get(rarity, 1.0)
    # center the glimpse inside the GUESS phase by default (you can modify GLIMPSE_POSITION_IN_GUESS)
    guess_start_time = intro_dur
    guess_end_time = intro_dur + GUESS_DUR
    # ensure glimpse fits inside GUESS
    glimpse_dur = min(glimpse_dur, GUESS_DUR)
    # position start so glimpse is centered at GLIMPSE_POSITION_IN_GUESS fraction of the GUESS duration
    glimpse_center = guess_start_time + GLIMPSE_POSITION_IN_GUESS * GUESS_DUR
    glimpse_start = max(guess_start_time, glimpse_center - 0.5 * glimpse_dur)
    glimpse_end = glimpse_start + glimpse_dur

    # For one-frame glimpses, compute the target frame index
    glimpse_start_frame = int(round(glimpse_start * fps))
    glimpse_end_frame = int(round(glimpse_end * fps))

    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        base = bg.copy()
        base.alpha_composite(overlay)

        # --- INTRO ---
        if include_intro and t < intro_dur:
            if cube_img:
                cube_center_y = int(H * 0.24)
                grow_p = ease_out_cubic(min(1.0, t / (intro_dur * 0.6))) if intro_dur > 0 else 1.0
                cube_scale = 0.6 + 0.4 * grow_p
                c_w = int(cube_img.width * cube_scale)
                c_h = int(cube_img.height * cube_scale)
                cube_resized = cube_img.resize((c_w, c_h), Image.LANCZOS)
                alpha = 1.0
                if t < 0.4:
                    alpha = ease_out_cubic(t / 0.4)
                elif t > intro_dur - 0.35:
                    alpha = max(0.0, (intro_dur - t) / 0.35)
                cube_to_paste = apply_alpha(cube_resized.copy(), alpha)
                cx = (W - c_w)//2
                cy = cube_center_y - c_h//2
                base.alpha_composite(cube_to_paste, dest=(cx, cy))
                if secret_card_img:
                    sc_grow_p = ease_out_cubic(min(1.0, t / (intro_dur * 0.6))) if intro_dur > 0 else 1.0
                    sc_scale = 0.7 + 0.3 * sc_grow_p
                    sc_w, sc_h = secret_card_img.size
                    new_sw = int(sc_w * sc_scale)
                    new_sh = int(sc_h * sc_scale)
                    secret_resized = secret_card_img.resize((new_sw, new_sh), Image.LANCZOS)
                    sc_alpha = 1.0
                    if t < 0.4:
                        sc_alpha = ease_out_cubic(t / 0.4)
                    elif t > intro_dur - 0.35:
                        sc_alpha = max(0.0, (intro_dur - t) / 0.35)
                    sc_to_paste = apply_alpha(secret_resized.copy(), sc_alpha)
                    scx = (W - new_sw)//2
                    scy = int(H * 0.76 - sc_h / 2)
                    base.alpha_composite(sc_to_paste, dest=(scx, scy))
            cix = (W - concept_img_intro.width)//2
            ciy = (H - concept_img_intro.height)//2
            base.alpha_composite(concept_img_intro, dest=(cix, ciy))

        # --- GUESS PHASE (with brief non-pixelated glimpse) ---
        elif t < intro_dur + GUESS_DUR:
            tt = t - intro_dur
            rem_dur = GUESS_DUR
            p = min(1.0, tt / min(0.8, rem_dur * 0.5))
            p_eased = ease_out_cubic(p)
            dy = int(bob_amp * math.sin(2 * math.pi * t / bob_period))

            # Animate a subtle entrance for the (brief) card graphic:
            # start off-screen bottom, slide to target (but may only be visible during glimpse window)
            card_w_actual, card_h_actual = full_card_img.size
            card_x = (W - card_w_actual)//2
            target_y = int(H * 0.5 - card_h_actual / 2) + dy
            start_y = H + 40
            card_y = int(start_y + (target_y - start_y) * p_eased)

            # decide whether to render the glimpse at this frame
            show_glimpse = (frame_idx >= glimpse_start_frame) and (frame_idx < glimpse_end_frame)
            if show_glimpse:
                # apply configured opacity
                card_to_paste = apply_alpha(glimpse_img.copy(), glimpse_opacity)
                base.alpha_composite(card_to_paste, dest=(card_x, card_y))

            # subtext at bottom during guess
            six = (W - sub_img.width)//2
            siy = int(H * 0.88)
            base.alpha_composite(sub_img, dest=(six, siy))

            # ---------- COUNTDOWN overlay (on top of the card area) ----------
            COUNTDOWN_DUR = 3  # total time for 3,2,1 (~1s each here)
            time_to_reveal = (intro_dur + GUESS_DUR) - t
            if 0 < time_to_reveal <= COUNTDOWN_DUR:
                numbers = ["3", "2", "1"]
                seg = COUNTDOWN_DUR / 3.0
                num_index = int((COUNTDOWN_DUR - time_to_reveal) / seg)
                num_index = min(num_index, 2)
                current_number = numbers[num_index]

                countdown_font_path = "Fonts/Clash_Bold.otf"
                countdown_img = pil_text_image(
                    current_number,
                    countdown_font_path,
                    260,
                    color=TEXT_COLOR,
                    max_width=W
                )

                local_t = ((COUNTDOWN_DUR - time_to_reveal) % seg) / seg
                fade_in = ease_out_cubic(min(1, local_t * 3))
                fade_out = 1.0 - max(0, (local_t - 0.6) * 2.5)
                alpha = max(0, min(1, fade_in * fade_out))

                scale = 0.8 + 0.25 * ease_out_cubic(fade_in)
                w_scaled = int(countdown_img.width * scale)
                h_scaled = int(countdown_img.height * scale)
                countdown_resized = countdown_img.resize((w_scaled, h_scaled), Image.LANCZOS)
                countdown_resized = apply_alpha(countdown_resized.copy(), alpha)

                cx = (W - w_scaled) // 2
                cy = card_y + (card_h_actual // 2) - (h_scaled // 2)
                base.alpha_composite(countdown_resized, dest=(cx, cy))

        # --- REVEAL PHASE: either normal reveal OR subscribe-outro ---
        else:
            tt = t - (intro_dur + GUESS_DUR)
            rem_dur = reveal_dur
            p = min(1.0, tt / max(0.001, rem_dur * 0.4))
            p_eased = ease_out_cubic(p)
            dy = int(bob_amp * math.sin(2 * math.pi * t / bob_period))

            if outro_subscribe:
                if thumbs_img:
                    thumb_w, thumb_h = thumbs_img.size
                    target_y = int(H * 0.48 - thumb_h / 2) + dy
                    start_y = -thumb_h - 40
                    thumb_y = int(start_y + (target_y - start_y) * p_eased)
                    thumb_x = (W - thumb_w)//2
                    thumb_to_paste = apply_alpha(thumbs_img.copy(), 1.0)
                    base.alpha_composite(thumb_to_paste, dest=(thumb_x, thumb_y))
                cix = (W - subscribe_img.width)//2
                ciy = int(H * 0.72)
                base.alpha_composite(subscribe_img, dest=(cix, ciy))
            else:
                card_w_actual, card_h_actual = full_card_img.size
                card_x = (W - card_w_actual)//2
                target_y = int(H * 0.5 - card_h_actual / 2) + dy
                start_y = int(H * 0.45)
                card_y = int(start_y + (target_y - start_y) * p_eased)
                alpha_card = 1.0
                card_to_paste = apply_alpha(full_card_img.copy(), alpha_card)
                base.alpha_composite(card_to_paste, dest=(card_x, card_y))
                nix = (W - name_img.width)//2
                niy = card_y + card_h_actual + 30
                base.alpha_composite(name_img, dest=(nix, niy))

        # --- TITLE: draw last so it is on top BUT only during GUESS+REVEAL ---
        if t >= intro_dur:
            tx = (W - title_img.width)//2
            ty = int(H * 0.07)
            base.alpha_composite(title_img, dest=(tx, ty))

        final = base
        bgr = pil_to_bgr(final)
        writer.write(bgr)
        if frame_idx % fps == 0 or frame_idx == total_frames - 1:
            print(f"  frame {frame_idx+1}/{total_frames} (t={t:.2f}s)")
    writer.release()
    print("Wrote:", outpath)

# ---------- AUDIO TIMING / MIXING ----------
def _find_sound(sounds_dir, basename_no_ext):
    exts = [".mp3", ".ogg", ".wav", ".m4a"]
    for e in exts:
        p = os.path.join(sounds_dir, basename_no_ext + e)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(sounds_dir, basename_no_ext + ".*"))
    return g[0] if g else None

def _find_random_happy(sounds_dir):
    candidates = glob.glob(os.path.join(sounds_dir, "Happy*.*"))
    return random.choice(candidates) if candidates else None

def add_timed_sounds(video_path, out_path, reveal_time, outro_subscribe=False, sounds_dir="Audios", ffmpeg_path="ffmpeg", ffprobe_path="ffprobe"):
    sounds = {}
    sounds['count3'] = _find_sound(sounds_dir, '3')
    sounds['count2'] = _find_sound(sounds_dir, '2')
    sounds['count1'] = _find_sound(sounds_dir, '1')
    sounds['valid'] = _find_sound(sounds_dir, 'Valid')
    sounds['win'] = _find_sound(sounds_dir, 'Win')
    sounds['happy'] = _find_random_happy(sounds_dir)
    sounds['intro'] = _find_sound(sounds_dir, "Intro")

    if outro_subscribe and sounds['win']:
        celebration = sounds['win']
    else:
        celebration = sounds['happy']

    sound_inputs = []
    if sounds.get('intro'):
        sound_inputs.append(('intro', sounds['intro'], 'intro'))
    for k in ['count3', 'count2', 'count1']:
        if sounds.get(k):
            sound_inputs.append(('count', sounds[k], k))
    if sounds.get('valid'):
        sound_inputs.append(('valid', sounds['valid'], 'valid'))
    if celebration:
        sound_inputs.append(('celebrate', celebration, 'celebrate'))

    if not sound_inputs:
        print("No timed sounds to add; skipping audio mixing.")
        try:
            os.replace(video_path, out_path)
        except Exception:
            subprocess.check_call(["cp", video_path, out_path])
        return

    try:
        subprocess.run([ffmpeg_path, "-version"], capture_output=True)
    except Exception:
        print("ffmpeg not found on PATH — cannot add timed sounds. Skipping audio step.")
        try:
            os.replace(video_path, out_path)
        except Exception:
            subprocess.check_call(["cp", video_path, out_path])
        return

    delays_ms = []
    for tag, path, key in sound_inputs:
        if tag == 'intro':
            start = 0.0
            delays_ms.append((path, int(round(start * 1000))))
        if tag == 'count':
            if key == 'count3':
                start = reveal_time - 3.0
            elif key == 'count2':
                start = reveal_time - 2.0
            else:
                start = reveal_time - 1.0
            start = max(0.0, start)
            delays_ms.append((path, int(round(start * 1000))))
        elif tag == 'valid':
            try:
                cmd = [ffprobe_path, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path]
                out = subprocess.run(cmd, capture_output=True, text=True)
                dur = float(out.stdout.strip()) if out.stdout.strip() else 0.0
            except Exception:
                dur = 0.0
            start = reveal_time
            start = max(0.0, start)
            delays_ms.append((path, int(round(start * 1000))))
        elif tag == 'celebrate':
            start = max(0.0, reveal_time)
            delays_ms.append((path, int(round(start * 1000))))

    cmd = [ffmpeg_path, '-y']
    cmd += ['-i', video_path]
    for p, d in delays_ms:
        cmd += ['-i', p]

    filter_parts = []
    s_labels = []
    for idx, (p, d) in enumerate(delays_ms, start=1):
        filter_parts.append(f"[{idx}:a]adelay={d}|{d},apad[s{idx}]")
        s_labels.append(f"[s{idx}]")

    has_audio = False
    try:
        cmd_probe = [ffprobe_path, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', video_path]
        p = subprocess.run(cmd_probe, capture_output=True, text=True)
        has_audio = bool(p.stdout.strip())
    except Exception:
        has_audio = True

    if has_audio:
        mix_inputs = ['[0:a]'] + s_labels
        mix_count = len(mix_inputs)
        filter_complex = ';'.join(filter_parts) + ';' + ''.join(mix_inputs) + f"amix=inputs={mix_count}:duration=longest:dropout_transition=0[aout]"
        cmd += ['-filter_complex', filter_complex, '-map', '0:v:0', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out_path]
    else:
        mix_count = len(s_labels)
        filter_complex = ';'.join(filter_parts) + ';' + ''.join(s_labels) + f"amix=inputs={mix_count}:duration=longest[aout]"
        cmd += ['-filter_complex', filter_complex, '-map', '0:v:0', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out_path]

    try:
        print('Running ffmpeg for timed sounds:')
        print(' '.join(cmd))
        subprocess.check_call(cmd)
        print('Added timed sounds — wrote:', out_path)
    except subprocess.CalledProcessError as e:
        print('ffmpeg failed while mixing timed sounds:', e)
        try:
            os.replace(video_path, out_path)
        except Exception:
            subprocess.check_call(["cp", video_path, out_path])

# Generate for a rarity (multiple cards)
def generate_for_rarity(rarity, used_cards):
    print("Generating:", rarity)
    backgrounds = list_images(BACKGROUND_DIR)
    if not backgrounds:
        raise RuntimeError("No background found in " + BACKGROUND_DIR)
    candidates = [b for b in backgrounds if rarity.lower() in os.path.basename(b).lower()]
    bg = candidates[0] if candidates else backgrounds[0]
    cards = [c for c in list_images(CARDS_DIR) if c not in used_cards]
    if not cards:
        raise RuntimeError("No unused cards left for " + rarity)
    count = COUNT_BY_RARITY.get(rarity, 1)
    chosen = random.sample(cards, min(count, len(cards)))
    used_cards.update(chosen)

    for idx, card in enumerate(chosen):
        include_intro = (idx == 0)
        outro_subscribe = (rarity == "Champion" and idx == 0)
        outname = os.path.join(OUT_DIR, f"guess_{rarity.lower()}_{idx+1}.mp4")

        write_card_video(card, bg, ITEMS_DIR, rarity, outname,
                        duration=DURATION_PER_CARD, fps=FPS, video_size=VIDEO_SIZE,
                        include_intro=include_intro, outro_subscribe=outro_subscribe)

        reveal_time = (INTRO_DUR if include_intro else 0.0) + GUESS_DUR
        out_with_sound = outname.replace('.mp4', '_sound.mp4')
        try:
            add_timed_sounds(outname, out_with_sound, reveal_time, outro_subscribe=outro_subscribe, sounds_dir='Audios')
            os.replace(out_with_sound, outname)
        except Exception as e:
            print('Failed to add timed sounds:', e)

def get_video_duration_seconds(path, fps=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video to get duration: " + path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps is None:
        fps_val = cap.get(cv2.CAP_PROP_FPS) or FPS
    else:
        fps_val = fps
    cap.release()
    if fps_val <= 0:
        raise RuntimeError("Invalid FPS value when computing duration")
    duration = float(frame_count) / float(fps_val)
    return duration

def mux_audio_with_ffmpeg(video_path, audio_path, out_path, loop_audio=False, ffmpeg_path="ffmpeg"):
    duration = get_video_duration_seconds(video_path)
    duration_str = f"{duration:.3f}"
    has_audio = has_audio_ffprobe(video_path)
    cmd = [ffmpeg_path, "-y"]
    if loop_audio:
        cmd += ["-stream_loop", "-1", "-i", audio_path]
    else:
        cmd += ["-i", audio_path]
    cmd += ["-i", video_path]
    if has_audio:
        filter_complex = (
            "[0:a]volume=0.40[a_music];"
            "[1:a]volume=1.00[a_video];"
            "[a_music][a_video]amix=inputs=2:duration=longest:dropout_transition=2[aout]"
        )
        cmd += [
            "-t", duration_str,
            "-filter_complex", filter_complex,
            "-map", "1:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_path
        ]
    else:
        cmd += [
            "-t", duration_str,
            "-map", "1:v:0",
            "-map", "0:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_path
        ]
    try:
        print("Running ffmpeg:", " ".join(cmd))
        subprocess.check_call(cmd)
        print("Wrote video-with-audio:", out_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg failed: " + str(e))

# ---------- CONCAT ----------
def has_audio_ffprobe(path, ffprobe_path="ffprobe"):
    try:
        cmd = [ffprobe_path, "-v", "error", "-select_streams", "a", "-show_entries",
               "stream=index", "-of", "csv=p=0", path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return bool(p.stdout.strip())
    except Exception:
        return True

def concatenate_videos(video_paths, outpath, ffmpeg_path="ffmpeg", ffprobe_path="ffprobe"):
    import os, math, shlex
    if not video_paths:
        raise ValueError("No videos to concatenate")
    cmd = [ffmpeg_path, "-y"]
    video_input_indices = []
    audio_input_indices = []
    current_input_index = 0
    for vp in video_paths:
        vp_abs = os.path.abspath(vp)
        cmd += ["-i", vp_abs]
        video_input_indices.append(current_input_index)
        current_input_index += 1
        if has_audio_ffprobe(vp_abs, ffprobe_path=ffprobe_path):
            audio_input_indices.append(current_input_index - 1)
        else:
            dur = get_video_duration_seconds(vp_abs)
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000"]
            audio_input_indices.append(current_input_index)
            current_input_index += 1

    pairs = []
    for vi, ai in zip(video_input_indices, audio_input_indices):
        pairs.append(f"[{vi}:v:0][{ai}:a:0]")

    n = len(video_paths)
    filter_complex = "".join(pairs) + f"concat=n={n}:v=1:a=1[outv][outa]"

    cmd += ["-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            outpath]

    print("Running ffmpeg concat (filter) with command:")
    print(" ".join(shlex.quote(p) for p in cmd))
    try:
        subprocess.check_call(cmd)
        print("Concatenation finished — wrote:", outpath)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg concat (filter) failed: " + str(e))

# ---------- MAIN ----------
if __name__ == "__main__":
    ensure_out()
    rarities = ["Common", "Rare", "Epic", "Legendary", "Champion"]
    used_cards = set()

    for r in rarities:
        try:
            generate_for_rarity(r, used_cards)
        except Exception as e:
            print("Error for", r, e)

    video_files_ordered = []
    for r in rarities:
        prefix = f"guess_{r.lower()}_"
        files = [f for f in os.listdir(OUT_DIR) if f.startswith(prefix) and f.lower().endswith(".mp4")]
        files.sort()
        files_full = [os.path.join(OUT_DIR, f) for f in files]
        video_files_ordered.extend(files_full)
    if video_files_ordered:
        final_out = os.path.join(OUT_DIR, "combined_guess_cards.mp4")
        concatenate_videos(video_files_ordered, final_out)
        print("All done — combined video at:", final_out)

        audio_in = "Audios/OvertimeMusic.mp3"
        final_with_audio = os.path.join(OUT_DIR, "combined_guess_cards_audio.mp4")
        mux_audio_with_ffmpeg(final_out, audio_in, final_with_audio, loop_audio=False)
    else:
        print("No videos produced to concatenate.")

    print("\n=== Set of all cards used ===")
    for card in sorted(used_cards):
        print(card)