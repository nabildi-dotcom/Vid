import json, os, sys, numpy as np
from faster_whisper import WhisperModel
a = np.fromfile('work/audio16k.raw', dtype=np.int16).astype(np.float32) / 32768.0
m = WhisperModel('small', device='cpu', compute_type='int8', download_root='models/whisper')
segs, info = m.transcribe(a, language=sys.argv[1] if len(sys.argv) > 1 else None, word_timestamps=True, beam_size=5, vad_filter=False)
words = []
for s in segs:
    for w in s.words:
        words.append({'word': w.word.strip(), 'start': round(w.start, 3), 'end': round(w.end, 3), 'prob': round(w.probability, 2)})
print(info.language, info.language_probability)
json.dump(words, open('work/words_raw.json', 'w'), ensure_ascii=False, indent=1)
for w in words: print(f"{w['start']:6.2f} {w['end']:6.2f} {w['prob']:.2f} {w['word']}")
