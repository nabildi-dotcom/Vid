import os, sys, time
os.environ.setdefault('U2NET_HOME', os.path.abspath('models/rembg'))
import onnxruntime as ort
_SO = ort.SessionOptions
def _lean_opts():
    o = _SO()
    o.enable_cpu_mem_arena = False   # give memory back after each frame
    o.enable_mem_pattern = False
    return o
ort.SessionOptions = _lean_opts
from rembg import new_session, remove
from PIL import Image
start, end = int(sys.argv[1]), int(sys.argv[2])
os.makedirs('work/mask', exist_ok=True)
sess = new_session('birefnet-portrait')
t0 = time.time()
for n in range(start, end + 1):
    out = f'work/mask/{n:04d}.png'
    if os.path.exists(out): continue
    m = remove(Image.open(f'work/frames/{n:04d}.png'), session=sess, only_mask=True, post_process_mask=False)
    m.save(out)
    print(n, f'{time.time()-t0:.1f}s', flush=True)
