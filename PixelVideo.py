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
PIXEL_BY_RARITY = {
    "Common": 40,
    "Rare": 60,
    "Epic": 80,
    "Legendary": 100,
    "Champion": 120
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

def pixelate_pil(img_path, max_size, pixel_block):
    """
    Pixelate image while preserving aspect ratio.
    max_size: (max_w, max_h) - the bounding box the image must fit inside.
    pixel_block: integer representing the desired block size in final pixels
                 (larger -> blockier)
    Returns an RGBA image sized to fit within max_size (no stretching), pixelated,
    with consistent block size independent of source resolution.
    """
    im = Image.open(img_path).convert("RGBA")
    orig_w, orig_h = im.size
    max_w, max_h = max_size
    # compute target size preserving aspect ratio to fit within max_size
    if orig_w == 0 or orig_h == 0:
        return im.resize((max_w, max_h), Image.LANCZOS)
    ratio = min(max_w / orig_w, max_h / orig_h)
    target_w = max(1, int(orig_w * ratio))
    target_h = max(1, int(orig_h * ratio))
    if pixel_block <= 1:
        return im.resize((target_w, target_h), Image.LANCZOS)
    # To produce consistent blockiness independent of source resolution:
    # downscale to (target_w // pixel_block, target_h // pixel_block),
    # then upscale to target size using NEAREST.
    small_w = max(1, target_w // pixel_block)
    small_h = max(1, target_h // pixel_block)
    # Use bilinear to downscale for smoother aggregated color then nearest to enlarge as blocks
    small = im.resize((small_w, small_h), resample=Image.BILINEAR)
    big = small.resize((target_w, target_h), Image.NEAREST)
    return big

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
    outro_subscribe: if True, the REVEAL PHASE is replaced by an intro-like "subscribe" scene
                     using Items/Thumbs.png (or any file containing 'thumbs').
    """
    W, H = video_size
    total_frames = int(math.ceil(duration * fps))
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
    # pixelised card: compute the max bounding box (no stretching)
    pixel_block = PIXEL_BY_RARITY.get(rarity, 12)
    max_card_h = int(H * 0.45)
    max_card_w = int(max_card_h * 0.65)
    pixel_card_img = pixelate_pil(card_path, (max_card_w, max_card_h), pixel_block)
    # full (revealed) card image, aspect-preserving
    full_card_img = fit_image_preserve(card_path, (max_card_w, max_card_h))
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
    # Make it larger by default: THUMBS_MAX_WIDTH_RATIO controls proportion of video width (0.0-1.0)
    THUMBS_MAX_WIDTH_RATIO = 0.85   # fraction of video width the thumbs should occupy
    THUMBS_ALLOW_UPSCALE = True     # allow enlarging small source thumbs images
    THUMBS_MAX_UPSCALE = 3.0        # don't scale a tiny 10px icon to > 3x its size (adjust as desired)

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
                    # desired scale to make thumbs width = tmax_w
                    desired_scale = float(tmax_w) / float(tw)
                    if not THUMBS_ALLOW_UPSCALE:
                        # behave like before: never enlarge (only shrink)
                        scale = min(1.0, desired_scale)
                    else:
                        # allow upscaling but clamp it to a maximum factor
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

    # concept text for INTRO (always the original "Can you guess..." — remains unchanged)
    concept_text_intro = f"Can you guess the PIXELATED card? BUT it gets HARDER.."
    concept_img_intro = pil_text_image(concept_text_intro, FONT_PATH, CONCEPT_FONT_SIZE, color=TEXT_COLOR, max_width=int(W*0.95))

    # subscribe text for REVEAL (only used in reveal when outro_subscribe True)
    subscribe_text = "Subscribe for more Clash Royale tests! Give your ANSWER in the comments!"
    subscribe_img = pil_text_image(subscribe_text, FONT_PATH, CONCEPT_FONT_SIZE - 6, color=TEXT_COLOR, max_width=int(W*0.95), align="center")

    # card name (filename without extension) for reveal (may be unused if outro_subscribe True)
    base_name = os.path.splitext(os.path.basename(card_path))[0]
    name_img = pil_text_image(base_name, FONT_PATH, NAME_FONT_SIZE, color=TEXT_COLOR, max_width=int(W*0.9))

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(outpath, fourcc, fps, (W, H))
    # animation params
    bob_amp = 8
    bob_period = 2.5

    # If intro is omitted, treat intro_dur as 0 so GUESS/REVEAL start immediately.
    intro_dur = INTRO_DUR if include_intro else 0.0

    if outro_subscribe:
        reveal_dur = OUTRO_SUBSCRIBE_DUR
    else:
        reveal_dur = REVEAL_DUR
    duration = intro_dur + GUESS_DUR + reveal_dur

    total_frames = int(math.ceil(duration * fps))

    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        base = bg.copy()
        base.alpha_composite(overlay)
        # --- INTRO PHASE (cube + secret card + concept text). Logo removed here ---
        if include_intro and t < intro_dur:
            # place cube higher in the frame so title area (top) stays clear
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
                # --- SECRET CARD (animated like cube) ---
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
            # --- IMPORTANT: keep the intro text the same as original ---
            cix = (W - concept_img_intro.width)//2
            ciy = (H - concept_img_intro.height)//2
            base.alpha_composite(concept_img_intro, dest=(cix, ciy))

        elif t < intro_dur + GUESS_DUR:
            tt = t - intro_dur
            rem_dur = GUESS_DUR
            p = min(1.0, tt / min(0.8, rem_dur * 0.5))
            p_eased = ease_out_cubic(p)
            dy = int(bob_amp * math.sin(2 * math.pi * t / bob_period))
            card_w_actual, card_h_actual = pixel_card_img.size
            card_x = (W - card_w_actual)//2
            target_y = int(H * 0.5 - card_h_actual / 2) + dy
            start_y = H + 40
            card_y = int(start_y + (target_y - start_y) * p_eased)
            alpha_card = p_eased
            card_to_paste = apply_alpha(pixel_card_img.copy(), alpha_card)
            base.alpha_composite(card_to_paste, dest=(card_x, card_y))

            # subtext at bottom during guess
            six = (W - sub_img.width)//2
            siy = int(H * 0.88)
            base.alpha_composite(sub_img, dest=(six, siy))

            # ---------- COUNTDOWN overlay (on top of the pixelated card) ----------
            COUNTDOWN_DUR = 3  # total time for 3,2,1 (~0.6s each)
            time_to_reveal = (intro_dur + GUESS_DUR) - t
            if 0 < time_to_reveal <= COUNTDOWN_DUR:
                numbers = ["3", "2", "1"]
                seg = COUNTDOWN_DUR / 3.0
                num_index = int((COUNTDOWN_DUR - time_to_reveal) / seg)
                num_index = min(num_index, 2)
                current_number = numbers[num_index]

                # Load bold font for countdown
                countdown_font_path = "Fonts/Clash_Bold.otf"
                countdown_img = pil_text_image(
                    current_number,
                    countdown_font_path,
                    260,
                    color=TEXT_COLOR,
                    max_width=W
                )

                # Smooth fade + pop animation
                local_t = ((COUNTDOWN_DUR - time_to_reveal) % seg) / seg
                fade_in = ease_out_cubic(min(1, local_t * 3))
                fade_out = 1.0 - max(0, (local_t - 0.6) * 2.5)
                alpha = max(0, min(1, fade_in * fade_out))

                # Scale pop (gentle bounce)
                scale = 0.8 + 0.25 * ease_out_cubic(fade_in)
                w_scaled = int(countdown_img.width * scale)
                h_scaled = int(countdown_img.height * scale)
                countdown_resized = countdown_img.resize((w_scaled, h_scaled), Image.LANCZOS)
                countdown_resized = apply_alpha(countdown_resized.copy(), alpha)

                # Center directly on top of the card
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
                # animate the thumbs image similarly to intro / card sliding
                if thumbs_img:
                    thumb_w, thumb_h = thumbs_img.size
                    # slide in from top (start above frame), land near center (adjust target_y as needed)
                    target_y = int(H * 0.48 - thumb_h / 2) + dy
                    start_y = -thumb_h - 40
                    thumb_y = int(start_y + (target_y - start_y) * p_eased)
                    thumb_x = (W - thumb_w)//2
                    thumb_to_paste = apply_alpha(thumbs_img.copy(), 1.0)
                    base.alpha_composite(thumb_to_paste, dest=(thumb_x, thumb_y))
                # subscribe text centered (ONLY in reveal when outro_subscribe is True)
                cix = (W - subscribe_img.width)//2
                ciy = int(H * 0.72)
                base.alpha_composite(subscribe_img, dest=(cix, ciy))
            else:
                # normal reveal: full card + name text
                card_w_actual, card_h_actual = full_card_img.size
                card_x = (W - card_w_actual)//2
                target_y = int(H * 0.5 - card_h_actual / 2) + dy
                start_y = int(H * 0.45)
                card_y = int(start_y + (target_y - start_y) * p_eased)
                alpha_card = 1.0
                card_to_paste = apply_alpha(full_card_img.copy(), alpha_card)
                base.alpha_composite(card_to_paste, dest=(card_x, card_y))
                # name under card
                nix = (W - name_img.width)//2
                niy = card_y + card_h_actual + 30
                base.alpha_composite(name_img, dest=(nix, niy))

        # --- TITLE: draw last so it is on top BUT only during GUESS+REVEAL ---
        if t >= intro_dur:
            tx = (W - title_img.width)//2
            ty = int(H * 0.07)
            base.alpha_composite(title_img, dest=(tx, ty))

        # final
        final = base
        bgr = pil_to_bgr(final)
        writer.write(bgr)
        if frame_idx % fps == 0 or frame_idx == total_frames - 1:
            print(f"  frame {frame_idx+1}/{total_frames} (t={t:.2f}s)")
    writer.release()
    print("Wrote:", outpath)

# ---------- AUDIO TIMING / MIXING ----------

def _find_sound(sounds_dir, basename_no_ext):
    """Find a sound by exact basename (without extension) in Audios; return None if not found."""
    exts = [".mp3", ".ogg", ".wav", ".m4a"]
    for e in exts:
        p = os.path.join(sounds_dir, basename_no_ext + e)
        if os.path.exists(p):
            return p
    # try glob as fallback
    g = glob.glob(os.path.join(sounds_dir, basename_no_ext + ".*"))
    return g[0] if g else None


def _find_random_happy(sounds_dir):
    candidates = glob.glob(os.path.join(sounds_dir, "Happy*.*"))
    return random.choice(candidates) if candidates else None


def add_timed_sounds(video_path, out_path, reveal_time, outro_subscribe=False, sounds_dir="Audios", ffmpeg_path="ffmpeg", ffprobe_path="ffprobe"):
    """
    Mix multiple timed sounds into a video:
      - countdown: play 3,2,1 at reveal_time-3, -2, -1
      - valid: if present, start so it *ends* exactly at reveal_time
      - celebration: play Happy* at reveal_time (or Win.* if outro_subscribe True)
    The function uses ffmpeg filter_complex with adelay + amix.
    """
    # Locate audio files
    sounds = {}
    sounds['count3'] = _find_sound(sounds_dir, '3')
    sounds['count2'] = _find_sound(sounds_dir, '2')
    sounds['count1'] = _find_sound(sounds_dir, '1')
    sounds['valid'] = _find_sound(sounds_dir, 'Valid')
    sounds['win'] = _find_sound(sounds_dir, 'Win')
    sounds['happy'] = _find_random_happy(sounds_dir)
    sounds['intro'] = _find_sound(sounds_dir, "Intro")

    # choose celebration sound
    if outro_subscribe and sounds['win']:
        celebration = sounds['win']
    else:
        celebration = sounds['happy']

    # build list of actual sound inputs (skip None)
    sound_inputs = []
    # countdown order: 3,2,1
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
        # simply copy video
        try:
            os.replace(video_path, out_path)
        except Exception:
            subprocess.check_call(["cp", video_path, out_path])
        return

    # Check ffmpeg/ffprobe presence
    try:
        subprocess.run([ffmpeg_path, "-version"], capture_output=True)
    except Exception:
        print("ffmpeg not found on PATH — cannot add timed sounds. Skipping audio step.")
        try:
            os.replace(video_path, out_path)
        except Exception:
            subprocess.check_call(["cp", video_path, out_path])
        return

    # Determine delays (ms) for each sound input relative to video start
    delays_ms = []
    for tag, path, key in sound_inputs:
        if tag == 'intro':
            start = 0.0  # play at the very beginning
            delays_ms.append((path, int(round(start * 1000))))
        if tag == 'count':
            # key is 'count3'/'count2'/'count1'
            if key == 'count3':
                start = reveal_time - 3.0
            elif key == 'count2':
                start = reveal_time - 2.0
            else:
                start = reveal_time - 1.0
            start = max(0.0, start)
            delays_ms.append((path, int(round(start * 1000))))
        elif tag == 'valid':
            # We want valid to END exactly at reveal_time. Query its duration via ffprobe
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
            # start exactly at reveal_time
            start = max(0.0, reveal_time)
            delays_ms.append((path, int(round(start * 1000))))

    # Build ffmpeg command: input 0 = video, inputs 1..N = sound files
    cmd = [ffmpeg_path, '-y']
    cmd += ['-i', video_path]
    for p, d in delays_ms:
        cmd += ['-i', p]

    # Build filter_complex: delay each audio input and label them s1,s2,... then mix with existing audio if present
    # inputs mapping: video is 0, audios are 1..n
    filter_parts = []
    s_labels = []
    for idx, (p, d) in enumerate(delays_ms, start=1):
        # adelay requires the number of channels delay separated by | for each channel; using single value repeats for channels
        filter_parts.append(f"[{idx}:a]adelay={d}|{d},apad[s{idx}]")
        s_labels.append(f"[s{idx}]")

    # Does video already have audio?
    has_audio = False
    try:
        cmd_probe = [ffprobe_path, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', video_path]
        p = subprocess.run(cmd_probe, capture_output=True, text=True)
        has_audio = bool(p.stdout.strip())
    except Exception:
        has_audio = True  # be conservative

    # Now mixing
    if has_audio:
        # include [0:a] + all s_labels in amix
        mix_inputs = ['[0:a]'] + s_labels
        mix_count = len(mix_inputs)
        filter_complex = ';'.join(filter_parts) + ';' + ''.join(mix_inputs) + f"amix=inputs={mix_count}:duration=longest:dropout_transition=0[aout]"
        # map video (0:v) and [aout]
        cmd += ['-filter_complex', filter_complex, '-map', '0:v:0', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out_path]
    else:
        # mix only the s_labels
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
        # fallback: copy original
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
        # If this is the Champion first clip (the "last videos" you mentioned),
        # replace reveal with a subscribe/outro scene that uses Items/Thumbs.png
        outro_subscribe = (rarity == "Champion" and idx == 0)
        outname = os.path.join(OUT_DIR, f"guess_{rarity.lower()}_{idx+1}.mp4")

        write_card_video(card, bg, ITEMS_DIR, rarity, outname,
                        duration=DURATION_PER_CARD, fps=FPS, video_size=VIDEO_SIZE,
                        include_intro=include_intro, outro_subscribe=outro_subscribe)

        # ---- NEW: Add reveal sound + countdown + valid + celebration ----
        reveal_time = (INTRO_DUR if include_intro else 0.0) + GUESS_DUR
        out_with_sound = outname.replace('.mp4', '_sound.mp4')
        try:
            add_timed_sounds(outname, out_with_sound, reveal_time, outro_subscribe=outro_subscribe, sounds_dir='Audios')
            # replace original with mixed audio
            os.replace(out_with_sound, outname)
        except Exception as e:
            print('Failed to add timed sounds:', e)


def get_video_duration_seconds(path, fps=None):
    # Use OpenCV to compute duration (works without ffprobe)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video to get duration: " + path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # prefer to use the FPS passed in or capture FPS
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
    """
    Mix background audio (audio_path) with any existing audio in video_path.
    - If the video already has audio, the function mixes music + existing audio
      (music is attenuated so it sits as background).
    - If the video has no audio, the music is simply added (and trimmed/looped to video length).
    - loop_audio: if True, the music will be looped to cover the video duration.
    """
    duration = get_video_duration_seconds(video_path)
    duration_str = f"{duration:.3f}"

    # Check whether the video has an audio stream (use ffprobe helper if available)
    has_audio = has_audio_ffprobe(video_path)

    # Build inputs: put music first (so it's [0:a]) and video second ([1:v] / [1:a] if present)
    cmd = [ffmpeg_path, "-y"]

    # Optionally loop music input
    if loop_audio:
        cmd += ["-stream_loop", "-1", "-i", audio_path]
    else:
        cmd += ["-i", audio_path]

    # video input (may already contain audio)
    cmd += ["-i", video_path]

    # If the input video has audio, mix music + video's audio
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
    """Return True if file has at least one audio stream. Uses ffprobe."""
    try:
        cmd = [ffprobe_path, "-v", "error", "-select_streams", "a", "-show_entries",
               "stream=index", "-of", "csv=p=0", path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return bool(p.stdout.strip())
    except Exception:
        # If ffprobe not present or fails, be conservative and assume audio exists
        return True


def concatenate_videos(video_paths, outpath, ffmpeg_path="ffmpeg", ffprobe_path="ffprobe"):
    """
    Concatenate multiple videos robustly using the concat filter.
    - Ensures each input has an audio stream by adding a silent audio input for clips without audio.
    - Re-encodes (libx264 + aac) so differing input codecs won't break the concat.
    """
    import os, math, shlex, tempfile

    if not video_paths:
        raise ValueError("No videos to concatenate")

    # Build ffmpeg args
    cmd = [ffmpeg_path, "-y"]
    # We'll keep track of input indices for video and audio streams per file
    video_input_indices = []
    audio_input_indices = []
    current_input_index = 0

    for vp in video_paths:
        vp_abs = os.path.abspath(vp)
        # add the video input
        cmd += ["-i", vp_abs]
        video_input_indices.append(current_input_index)
        current_input_index += 1

        # check audio presence
        if has_audio_ffprobe(vp_abs, ffprobe_path=ffprobe_path):
            # audio is the same input index
            audio_input_indices.append(current_input_index - 1)
        else:
            # add a silent audio input matching video duration
            # get duration using your helper (falls back to ffprobe if needed)
            dur = get_video_duration_seconds(vp_abs)
            # create a lavfi anullsrc as a separate input; duration must be specified
            # sample rate & channel layout chosen to be standard
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000"]
            audio_input_indices.append(current_input_index)
            current_input_index += 1

    # Build filter_complex: for each file we need [v][a] pairs in the correct input indices order
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

    # Gather generated files in the desired order:
    video_files_ordered = []
    for r in rarities:
        prefix = f"guess_{r.lower()}_"
        files = [f for f in os.listdir(OUT_DIR) if f.startswith(prefix) and f.lower().endswith(".mp4")]
        files.sort()  # sorts by index in the filename
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