import time, numpy as np
for mb in (64, 256, 1024):
    t = time.time()
    a = np.ones(mb * 1024 * 1024 // 4, dtype=np.float32)
    dt = time.time() - t
    print(f"{mb:5d} MB touched in {dt:7.2f} s -> {mb/dt:8.1f} MB/s", flush=True)
    del a
