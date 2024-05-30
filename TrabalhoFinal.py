import numpy as np
from scipy import fftpack, interpolate
import matplotlib.pyplot as plt
from PIL import Image

# Parameters
original = "nome_da_foto"
alterado = original + "_alterado"
diferenca = original + "_diff"


# System parameters
wavelength = 632.8 * 10**(-9)  # m
f = 27.5  # focal length in [m]
system_size = 0.01  # in [m]
laser_size = 0.008  # in [m]
N_mask = 1024
pixsize_mask = system_size / N_mask

x = np.linspace(-system_size / 2, system_size / 2, num=N_mask)
y = np.linspace(-system_size / 2, system_size / 2, num=N_mask)
xx, yy = np.meshgrid(x, y)

# Frequencies in the Fourier plane
freq_x = fftpack.fftshift(fftpack.fftfreq(N_mask, d=pixsize_mask))
freq_y = fftpack.fftshift(fftpack.fftfreq(N_mask, d=pixsize_mask))
x_freq = freq_x * wavelength * f
y_freq = freq_y * wavelength * f
xx_freq, yy_freq = np.meshgrid(x_freq, y_freq)

# Input: monochromatic laser field with a given circular aperture
input_field = np.zeros((N_mask, N_mask))
input_field[np.sqrt(xx**2 + yy**2) < laser_size / 2] = 1

# Show input field
plt.imshow(input_field, extent=[-system_size / 2, system_size / 2, -system_size / 2, system_size / 2])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title("Laser Intensity")
plt.show()

# Import the object from an image file (ATENÇÃO AO PATH)
mypath = f'/home/vasco04/Desktop/{original}.jpg'
img = Image.open(mypath)
plt.imshow(img)
plt.title('Original Image')
plt.show()

# Convert image to grayscale, resize and normalize
img = img.convert('L')
img = img.resize((len(x), len(y)))
img_array = np.asarray(img)
object_ = img_array / 255.0

# Apply the object to the input field
input_field = input_field * object_

# Apply a circular window to the input field to ensure no leakage outside the laser area
input_field[np.sqrt(xx**2 + yy**2) >= laser_size / 2] = 0

# Show input field after applying the object
plt.imshow(input_field, extent=[-system_size / 2, system_size / 2, -system_size / 2, system_size / 2])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title("Laser Intensity after object")
plt.show()

# Fourier transform of the input field
fourier_plane = fftpack.fftshift(fftpack.fft2(fftpack.ifftshift(input_field)))

frequencies = np.array([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 
    37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 
    49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 
    61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 
    73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 
    85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 
    97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 
    107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 
    117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 
    127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 
    137.0
])  # frequencies

frequencies = frequencies * 1000

mtf_values = np.array([
    0.985756, 0.971512, 0.957268, 0.943024, 0.928780, 0.914537, 
    0.900293, 0.886049, 0.871805, 0.857561, 0.843317, 0.829073, 0.814829, 
    0.800585, 0.786341, 0.772098, 0.759750, 0.747880, 0.736010, 0.724140, 
    0.712270, 0.700400, 0.688530, 0.676660, 0.664790, 0.652921, 0.641051, 
    0.629181, 0.617311, 0.605441, 0.593571, 0.581701, 0.570540, 0.559857, 
    0.549174, 0.538491, 0.527809, 0.517126, 0.506443, 0.495760, 0.485077, 
    0.474394, 0.463711, 0.453028, 0.442345, 0.431662, 0.420979, 0.410296, 
    0.401965, 0.397217, 0.392469, 0.387721, 0.382973, 0.378225, 0.373477, 
    0.368729, 0.364683, 0.362309, 0.359935, 0.357561, 0.355187, 0.352813, 
    0.350439, 0.348065, 0.345922, 0.344735, 0.343548, 0.342361, 0.341174, 
    0.339987, 0.338800, 0.337613, 0.336203, 0.332642, 0.329081, 0.325520, 
    0.321959, 0.318398, 0.314837, 0.311276, 0.307715, 0.301795, 0.295860, 
    0.289925, 0.283990, 0.278055, 0.272120, 0.266185, 0.260250, 0.254315, 
    0.248380, 0.242445, 0.236510, 0.230575, 0.224640, 0.218705, 0.212770, 
    0.205895, 0.198773, 0.191651, 0.184529, 0.177407, 0.170285, 0.163163, 
    0.156041, 0.148919, 0.141797, 0.134675, 0.127553, 0.120431, 0.113309, 
    0.106187, 0.099066, 0.093347, 0.088599, 0.083851, 0.079103, 0.074355, 
    0.069607, 0.064859, 0.060111, 0.055363, 0.050615, 0.045867, 0.041119, 
    0.036371, 0.031623, 0.026875, 0.022127, 0.018768, 0.017581, 0.016394, 
    0.015207, 0.014020, 0.012833, 0.011646, 0.010459
])  # MTF values

# Add MTF value for frequency 0 (should be 1)
frequencies = np.insert(frequencies, 0, 0)
mtf_values = np.insert(mtf_values, 0, 1.0)

# Interpolate the MTF data
mtf_interpolator = interpolate.interp1d(frequencies, mtf_values, bounds_error=False, fill_value=0)

# Calculate the radial frequencies in the Fourier plane
radial_freq = np.sqrt(xx_freq**2 + yy_freq**2)

# Apply the interpolated MTF to the radial frequencies
mtf = mtf_interpolator(radial_freq)

# Show the interpolated MTF
plt.imshow(mtf, extent=[-system_size / 2, system_size / 2, -system_size / 2, system_size / 2])
plt.xlabel('Frequency x')
plt.ylabel('Frequency y')
plt.title("Interpolated MTF")
plt.colorbar()
#(ATENÇÃO AO PATH)
plt.savefig(f'/home/vasco04/Desktop/MTF.jpg', dpi=300)
plt.show()

# Apply the MTF to the Fourier plane
fourier_plane *= mtf

# Inverse Fourier transform to get the output field
output_field = fftpack.ifftshift(fftpack.ifft2(fftpack.fftshift(fourier_plane)))

# Show results
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Fourier optics')

axs[1, 0].set_title('Output Field')
im_1 = axs[1, 0].imshow(np.abs(output_field), extent=[-system_size / 2, system_size / 2, -system_size / 2, system_size / 2])
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('y (m)')

axs[0, 0].set_title('Fourier Transform')
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('y (m)')
im_2 = axs[0, 0].imshow(np.abs(fourier_plane), extent=[x_freq.min(), x_freq.max(), y_freq.min(), y_freq.max()])

axs[0, 1].set_title('Fourier Transform with MTF')
im_3 = axs[0, 1].imshow(np.abs(fourier_plane) * mtf, extent=[x_freq.min(), x_freq.max(), y_freq.min(), y_freq.max()])
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('y (m)')

axs[1, 1].set_title('Difference between input and output')
im_4 = axs[1, 1].imshow(np.abs(output_field - input_field), extent=[-system_size / 2, system_size / 2, -system_size / 2, system_size / 2])
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('y (m)')

im_2.set_clim(0, 700)
im_3.set_clim(0, 700)

plt.subplots_adjust(left=0.085, bottom=0.068, right=0.93, top=0.88, wspace=0.486, hspace=0.245)
#(ATENÇÃO AO PATH)
plt.savefig(f'/home/vasco04/Desktop/{alterado}.jpg', dpi=300)
plt.show()
