import numpy as np
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift, ifftn, ifftshift 

if not os.path.isfile("subdata.npy"):
    subdata = np.genfromtxt("subdata modified.csv", dtype=np.complex64, delimiter=',')
    np.save("subdata", subdata)
else:
    subdata = np.load("subdata.npy")

subdata = np.reshape(subdata, (64,64,64,49), order='F')

L = 10
n = 64
x = np.linspace(-L, L, n+1)[0:n]
y = np.linspace(-L, L, n+1)[0:n]
z = np.linspace(-L, L, n+1)[0:n]
k = (2*np.pi / (2 * L))*(np.append(np.arange(0, n/2), np.arange(-n/2, 0)))
ks = fftshift(k)

tau = .05
k0=3
gauss_filter = np.exp(-tau * (np.linspace(-10, 10, 1000)-k0)**2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('k')
ax.set_ylabel('g(k)')
ax.plot(np.linspace(-10, 10, 1000), gauss_filter)
plt.title('Gaussian Filter with $\\tau$=.05, $k_0$=3')
plt.show()

X, Y, Z = np.meshgrid(x, y, z)
Kx, Ky, Kz = np.meshgrid(ks, ks, ks)

# Pre: Plot trajectory from noisy data
noisy_pos = np.zeros((49, 3))
for i in range(49):
    x_ind, y_ind, z_ind = np.unravel_index(np.argmax(np.abs(subdata[:,:,:,i])), subdata[:,:,:,i].shape)
    noisy_pos[i, :] = [X[x_ind, y_ind, z_ind], Y[x_ind, y_ind, z_ind], Z[x_ind, y_ind, z_ind]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)
plt.title('Position of Submarine Using Noisy Data', fontsize=20)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], noisy_pos[:, 2])
ax.scatter(noisy_pos[:, 0], noisy_pos[:, 1], noisy_pos[:, 2])
plt.show()

# 1: Determine frequency signatures

# Take the Fourier Transform over the 3 spatial axes
subdata_transformed = fftshift(fftn(subdata, axes=(0,1,2)))
subdata_transformed_mean = np.mean(subdata_transformed, axis=3)
subdata_transformed_normalized = subdata_transformed_mean / np.amax(subdata_transformed_mean)
x_ind, y_ind, z_ind = np.unravel_index(np.argmax(subdata_transformed_normalized), subdata_transformed_normalized.shape)
freq = np.array([X[x_ind, y_ind, z_ind], Y[x_ind, y_ind, z_ind], Z[x_ind, y_ind, z_ind]])
print(freq)

# Generate points to graph, skipping those that are below the threshold
def getXYZC(array, X, Y, Z, threshold):
    x = []
    y = []
    z = []
    c = []

    for xi in range(array.shape[0]):
        for yi in range(array.shape[1]):
            for zi in range(array.shape[2]):
                val = np.abs(array[xi, yi, zi])
                if val < threshold:
                    continue
                x.append(X[xi, yi, zi])
                y.append(Y[xi, yi, zi])
                z.append(Z[xi, yi, zi])
                c.append(val)
    
    return x,y,z,c

# Show frequencies that are strongest
x,y,z,c = getXYZC(subdata_transformed_normalized, Kx, Ky, Kz, .4)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x,y,z, c=np.abs(c), cmap=plt.cool())
ax.set_xlabel('$K_x$')
ax.set_ylabel('$K_y$')
ax.set_zlabel('$K_z$')
plt.title('Plot of Frequencies with values >.4 shown')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
fig.colorbar(img)
plt.show()

# 2: Determine path of submarine

# Generate and visualize Gaussian filter
width = .5
gauss = np.exp(-width * ((Kx - freq[0])**2 + (Ky - freq[1])**2 + (Kz - freq[2])**2))
x,y,z,c = getXYZC(gauss, Kx, Ky, Kz, .1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x,y,z, c=np.abs(c), cmap=plt.cool())
ax.set_xlabel('$K_x$')
ax.set_ylabel('$K_y$')
ax.set_zlabel('$K_z$')
plt.title('Plot of Gaussian Filter with Points >.5')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
fig.colorbar(img)
plt.show()

# Apply Gaussian filter to data and re-visualize
pos = np.zeros((49, 3))
for i in range(49):
    subdata_i_transformed = fftshift(fftn(subdata[:,:,:,i]))
    subdata_i_transformed_filtered = subdata_i_transformed * gauss
    subdata_filtered = np.abs(ifftn(ifftshift(subdata_i_transformed_filtered)))
    x_ind, y_ind, z_ind = np.unravel_index(np.argmax(subdata_filtered), subdata_filtered.shape)
    pos[i, :] = [X[x_ind, y_ind, z_ind], Y[x_ind, y_ind, z_ind], Z[x_ind, y_ind, z_ind]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Position of Submarine Using Denoised Data')
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
ax.scatter(pos[1:-1, 0], pos[1:-1, 1], pos[1:-1, 2])
ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], 'g*', s=64)
ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], 'r*', s=64)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Comparison of Trajectories found\n by Noisy and Denoised Data', y=1.08)
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="Denoised Position")
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], noisy_pos[:, 2], 'r', label="Noisy Position")
ax.scatter(noisy_pos[:, 0], noisy_pos[:, 1], noisy_pos[:, 2], 'r')
ax.legend(loc="upper right")
plt.show()

# 3: Determine coordinates to send subtracking aircraft
print(pos[:, 0:2])
np.savetxt("Sub Pos.csv", pos[:, 0:2].T, delimiter=',', fmt=R"%1.4f")