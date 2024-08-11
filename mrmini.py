from sktime.transformations.panel.rocket._minirocket_numba import _fit, _transform
import time
import numpy as np
import numba

numba.set_num_threads(8)

# Precompile
data = np.random.rand(10, 100).astype(np.float32)
mini = _fit(data)
_transform(data, mini)

# Run 1
# 8c, 857.288122177124 ms
data = np.random.rand(100, 10000).astype(np.float32)
start = time.time()
mini = _fit(data)
_transform(data, mini)
end = time.time()
print(end - start)

# Run 2
# 8c, 13082.681655883789 ms
data = np.random.rand(100, 100000).astype(np.float32)
start = time.time()
mini = _fit(data)
_transform(data, mini)
end = time.time()
print(end - start)

# Run 3
# 8c, 314905.8892726898 ms
data = np.random.rand(100, 1000000).astype(np.float32)
start = time.time()
mini = _fit(data)
_transform(data, mini)
end = time.time()
print(end - start)

