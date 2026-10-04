import numpy as np
import matplotlib.pyplot as plt

# Distances from 0 through 65536, inclusive.
distances = np.arange(65537)

# Rotation frequencies for the 64 pairs.
m = np.arange(64)
frequencies = 10000.0 ** (-m / 64)

# Each row contains the 64 pair contributions for one distance.
angles = distances[:, None] * frequencies[None, :]
attention = np.cos(angles).mean(axis=1)

plt.figure(figsize=(12, 5))
plt.plot(distances, attention, linewidth=0.5)
plt.xlabel("Positional distance |i − j|")
plt.ylabel("Unscaled attention score A(i, j)")
plt.title("RoPE positional similarity (d = 128, base = 10000)")
plt.xlim(0, 65536)
plt.grid(alpha=0.3)
plt.tight_layout()

from pathlib import Path
plt.savefig(Path(__file__).with_name("rope_attention.png"), dpi=300)
plt.show()