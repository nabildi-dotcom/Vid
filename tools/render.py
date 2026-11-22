"""Compose: original frame -> big word -> cut-out person -> captions. Pipes to ffmpeg."""
import json, os, sys, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

st = json.load(open('tools/style.json'))
words = json.load(open('work/words.json'))
FPS = 60
frames = sorted(os.listdir('work/frames'))
N = len(frames)
W, H = Image.open('work/frames/0000.png').size
big = next(w for w in words if w['big'])
BIGF = round(big['start'] * FPS)

def ink(text, font_path, size, color):
    """Render text RTL-shaped, cropped tightly to the letters."""
    f = ImageFont.truetype(font_path, size)
    pad = size
    l, t, r, b = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), text, font=f, direction='rtl', language='ar')
    im = Image.new('RGBA', (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=tuple(color) + (255,), direction='rtl', language='ar')
    return im.crop(im.getchannel('A').getbbox())

# ---- big word: size to 85% width ----
probe = ink(st['big_word'], st['big_font'], 400, st['big_color'])
size = int(400 * st['big_width_frac'] * W / probe.width)
word = ink(st['big_word'], st['big_font'], size, st['big_color'])
if word.width > W * st['big_width_frac']:
    word = word.resize((int(W * st['big_width_frac']), int(word.height * W * st['big_width_frac'] / word.width)), Image.LANCZOS)
ww, wh = word.size
wx = (W - ww) // 2

# ---- head top on the big-word frame, middle 40% only ----
m0 = np.asarray(Image.open(f'work/mask/{BIGF:04d}.png').convert('L')).astype(np.float32) / 255
mid = m0[:, int(W * 0.3):int(W * 0.7)]
rows = np.where((mid > 0.5).sum(1) > 3)[0]
head_top = int(rows[0])
wy = int(head_top - (1 - st['head_overlap']) * wh)   # head covers the lower 60% of the letters
wa = np.asarray(word.getchannel('A')).astype(np.float32) / 255
def cover(y):
    ys, ye = max(y, 0), min(y + wh, H)
    m = m0[ys:ye, wx:wx + ww]
    a = wa[ys - y:ye - y]
    return float((a * m).sum() / max(a.sum(), 1))
auto_c = cover(wy)
if st.get('big_offset_y', 0) == 0:
    while cover(wy) < st['min_cover'] and wy + wh < H * 0.8:
        wy += 4
wy += int(st.get('big_offset_y', 0) * H)
final_c = cover(wy)
print(f'head top {head_top}px; word {ww}x{wh} at y={wy}; head covers {final_c*100:.0f}% of letters (before lowering {auto_c*100:.0f}%)', file=sys.stderr)

POP = [0.80, 0.92, 1.01, 1.05, 1.03, 1.01]
def word_layer(k):
    s = POP[k] if k < len(POP) else 1.0
    s = min(s, W / ww)  # never wider than the frame
    if abs(s - 1) < 1e-6: return word, wx, wy
    im = word.resize((max(1, int(ww * s)), max(1, int(wh * s))), Image.LANCZOS)
    cx, cy = wx + ww / 2, wy + wh / 2
    return im, int(cx - im.width / 2), int(cy - im.height / 2)

# ---- captions ----
csize = int(W * st['cap_size_frac'])
cfont = ImageFont.truetype(st['cap_font'], csize)
spoken = [w for w in words if not w['big']]
groups, i = [], 0
for g in st['groups']:
    ws = spoken[i:i + len(g)]; i += len(g)
    assert [w['word'] for w in ws] == g, (g, [w['word'] for w in ws])
    groups.append(ws)
asc, desc = cfont.getmetrics()
line_h = asc + desc
band_top = int(st['cap_band'][0] * H)
cap_bottom = min(wy - int(0.01 * H), int(st['cap_band'][1] * H))
cap_y = band_top if st.get('cap_top_frac') is None else int(st['cap_top_frac'] * H)
cap_y = max(band_top, min(cap_y, cap_bottom - line_h))
space = int(cfont.getlength(' ') * 1.2)

def cap_layer(t):
    g = None
    for gi, ws in enumerate(groups):
        nxt = groups[gi + 1][0]['start'] if gi + 1 < len(groups) else 1e9
        if ws[0]['start'] <= t < nxt: g = ws
    if g is None: return None
    widths = [ImageDraw.Draw(Image.new('L', (1, 1))).textlength(w['word'], font=cfont, direction='rtl', language='ar') for w in g]
    total = sum(widths) + space * (len(g) - 1)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    shadow = Image.new('L', (W, H), 0)
    d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    x = (W + total) / 2   # right edge; Arabic reads right to left
    for w, wd in zip(g, widths):
        x -= wd
        if t >= w['start']:
            d.text((x, cap_y), w['word'], font=cfont, fill=tuple(st['cap_color']) + (255,), direction='rtl', language='ar')
            ds.text((x + 3, cap_y + 4), w['word'], font=cfont, fill=190, direction='rtl', language='ar')
        x -= space
    sh = shadow.filter(ImageFilter.GaussianBlur(7))
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    out.putalpha(sh)
    out.alpha_composite(layer)
    return out

def frame(n):
    t = n / FPS
    im = Image.open(f'work/frames/{n:04d}.png').convert('RGBA')
    if n >= BIGF:
        orig = im.copy()
        wl, x, y = word_layer(n - BIGF)
        im.alpha_composite(wl, (x, y)) if x >= 0 and y >= 0 else im.paste(wl, (x, y), wl)
        mask = Image.open(f'work/mask/{n:04d}.png').convert('L')
        im = Image.composite(orig, im, mask)
    c = cap_layer(t)
    if c is not None: im.alpha_composite(c)
    return im.convert('RGB')

if __name__ == '__main__':
    mode = sys.argv[1]
    os.makedirs('output', exist_ok=True)
    if mode == 'png':
        n = int(sys.argv[2]); frame(n).save(sys.argv[3]); print('saved', sys.argv[3])
    else:
        out = sys.argv[2]
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
               '-i', 'input/talking.mov', '-map', '0:v', '-map', '1:a', '-c:a', 'copy',
               '-vf', 'scale=in_range=pc:out_range=pc:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-color_range', 'pc', '-colorspace', 'bt709',
               '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', '-shortest', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for n in range(N):
            p.stdin.write(frame(n).tobytes())
        p.stdin.close(); p.wait(); print('saved', out, 'exit', p.returncode)
