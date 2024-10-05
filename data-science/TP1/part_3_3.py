import os
import numpy as np
import matplotlib.pyplot as plt

data_folder = "data"
files = os.listdir(data_folder)

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs = axs.flatten()

for i, path in enumerate(files):
    data = np.load(f"{data_folder}/{path}")["data"]
    cov = np.cov(data, rowvar=False)
    eigvals = np.linalg.eigvals(cov)
    sorted_eigenvalues = np.sort(eigvals)[::-1]  # Sort eigenvalues in descending order

    axs[i].plot(sorted_eigenvalues)
    axs[i].set_xlabel('Rank of Eigenvalue')
    axs[i].set_ylabel('Eigenvalue')
    axs[i].set_title(f"Eigenspectrum of {path}")
    axs[i].grid(True)

plt.savefig("part_3_3.png")
plt.show()
