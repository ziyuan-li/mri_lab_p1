import os
import numpy as np
import matplotlib.pyplot as plt

fid_path = r"C:\Li\Projects\tum_courses\mri_lab_p1\data\SpinEcho_055521_24092025\Series0002\CorrectedFID_slice0001.fid"

raw = np.fromfile(fid_path,'d')

n_phase = int(np.sqrt(len(raw)/2))
n_freq = n_phase

data_cplx_1d = raw[0::2] + 1j * raw[1::2]

kspace = np.reshape(data_cplx_1d, (n_phase, n_freq))

center_line = np.abs(kspace[n_phase // 2, :])

plt.figure()
plt.plot(center_line)
plt.xlabel("frequency-encode index")
plt.ylabel("magnitude")
plt.tight_layout()
out_png = os.path.join("./fig/center_line.png")
plt.savefig(out_png)
plt.show()

img1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace, axes=1), axis=1), axes=1)
img2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(img1, axes=0), axis=0), axes=0)

mag1 = np.abs(img1)
mag2 = np.abs(img2)
mag1  = np.rot90(mag1 , k=-1)
mag1 /= mag1.max() + 1e-12
mag2 /= mag2.max() + 1e-12
mag2 = np.rot90(mag2, k=-1)

img_ifft2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
mag_ifft2 = np.abs(img_ifft2)
mag_ifft2 /= mag_ifft2.max() + 1e-12
mag_ifft2= np.rot90(mag_ifft2, k=-1)

plt.figure()
plt.imshow(mag1, cmap="gray")
plt.title("After first 1D FFT")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(mag2, cmap="gray")
plt.title("Reconstructed image ")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(mag_ifft2, cmap="gray")
plt.title("Reconstructed image (IFFT2)")
plt.axis("off")
plt.tight_layout()
plt.show()
plt.title("Reconstructed image (IFFT2)")

out_png = os.path.join("./fig/reconstruction_water_phantom.png")
plt.imsave(out_png, mag2, cmap="gray")
out_png = os.path.join("./fig/reconstruction_water_phantom_after_1D_FFT.png")
plt.imsave(out_png, mag1, cmap="gray")
out_png = os.path.join("./fig/reconstruction_water_phantom_2D_FFT.png")
plt.imsave(out_png, mag_ifft2, cmap="gray")


