import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import matplotlib.pyplot as plt


# img creation 32x32
def create_image():
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    return image


# kernel cod
cuda_code = """
  /// device funkcija za racunanje energije pixela
  __device__ int calculatePixelEnergy(unsigned char* shared_image, int idx, int width)
  {
      int pixelEnergy = 0;
      if (threadIdx.x < width - 1)
      {
          int rightIdx = idx + 3;
          for (int c = 0; c < 3; ++c)
          {
              pixelEnergy += abs(shared_image[idx + c] - shared_image[rightIdx + c]);
          }
      }
      if (threadIdx.y < width - 1)
      {
          int bottomIdx = idx + width * 3;
          for (int c = 0; c < 3; ++c)
          {
              pixelEnergy += abs(shared_image[idx + c] - shared_image[bottomIdx + c]);
          }
      }
      return pixelEnergy;
  }

  /// glavna global funkcija
  __global__ void calculateEnergy(unsigned char* image, int width, int height, int* energy)
  {
      extern __shared__ unsigned char shared_image[];

      /// kopiraj u shared mem
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      int idx = (y * width + x) * 3;
      if (x < width && y < height)
      {
          for (int c = 0; c < 3; ++c)
          {
              shared_image[idx + c] = image[idx + c];
          }
      }

      __syncthreads();

      /// Energy calculation
      if (x < width && y < height)
      {
          int pixelEnergy = calculatePixelEnergy(shared_image, idx, width);
          energy[threadIdx.y * blockDim.x + threadIdx.x] = pixelEnergy;
      }

      __syncthreads();

      /// reduce sa atomicadd
      if (threadIdx.x == 0 && threadIdx.y == 0)
      {
          int blockEnergy = 0;
          for (int i = 0; i < blockDim.x * blockDim.y; ++i)
          {
              blockEnergy += energy[i];
          }
          atomicAdd(&energy[0], blockEnergy);
      }
  }
"""


# povezivanje kernel koda na source mdoel
model = SourceModule(cuda_code)
calculate_energy = model.get_function("calculateEnergy")

# mem aloc za slku
image = create_image()
image_gpu = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(image_gpu, image)

shared_mem = 32 * 32 * 3 * np.dtype(np.uint8).itemsize
energy_device = gpuarray.zeros((1,), dtype=np.int32)

block_size = (32, 32, 1)
grid_size = (1, 1)

# pozivanje kernel funkcije
calculate_energy(image_gpu, np.int32(32), np.int32(32), energy_device, block=block_size, grid=grid_size, shared=shared_mem)

energy_host = energy_device.get()

# plotovanje slike
title = "Energy: " + str(energy_host[0])
plt.title(title)
plt.imshow(image)
plt.axis('off')
plt.show()

# print("Energy:", energy_host[0])


