import _cuSZp as cusz
import numpy as np

num_elements = 10
tol = 0.01
a = np.random.rand(num_elements)
compressed = cusz.compress(a,tol)
print(f"compressed bytes : {len(compressed)}")
decompressed = cusz.decompress(compressed,num_elements,tol)

print(a)
print(decompressed)

