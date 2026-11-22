import json, numpy as np
raw = json.load(open('work/words_raw.json'))
# index in raw transcript -> corrected word (None = dropped, unclear)
fix = {0:'كاين',1:None,5:'تخلصش',6:'والو',7:'وكاين',11:'تخلص',12:'غادي',13:'تخلص',14:'الضريبة',15:None,16:'الضريبة',17:'عليها',18:'تخلص',19:None,20:None}
BIG_INDEX = 14
a = np.fromfile('work/audio16k.raw', dtype=np.int16).astype(np.float32) / 32768.0
hop = 160  # 10 ms
rms = np.sqrt(np.convolve(a**2, np.ones(400)/400, 'same')[::hop])
db = 20*np.log10(rms + 1e-6)
floor = np.percentile(db, 10)
out = []
for i, w in enumerate(raw):
    text = fix.get(i, w['word'])
    if text is None: continue
    s, e = w['start'], w['end']
    i0, i1 = int(s*100), int(min(e - 0.05, s + 0.30)*100)
    i1 = max(i1, i0 + 1)
    seg = db[max(i0-5,0):int(e*100)+1]
    peak = seg.max()
    thr = max(floor + 10, peak - 15)
    new = s
    if db[i0] < thr:  # quiet at the given start: move to where it gets loud
        for k in range(i0, int((e - 0.05) * 100) + 1):
            if db[k] >= thr:
                new = k / 100; break
    else:  # look for the dip between words just after the start, then the rise
        win = db[i0:i1 + 1]
        d = int(np.argmin(win))
        if d > 0 and db[i0] - win[d] >= 4:
            for k in range(i0 + d, i1 + 1):
                if db[k] >= win[d] + 6:
                    new = k / 100; break
    out.append({'word': text, 'start': round(new, 3), 'end': e, 'orig_start': s, 'big': i == BIG_INDEX})
json.dump(out, open('work/words.json', 'w'), ensure_ascii=False, indent=1)
for w in out: print(f"{w['orig_start']:5.2f} -> {w['start']:5.2f}  {'*' if w['big'] else ' '} {w['word']}")
