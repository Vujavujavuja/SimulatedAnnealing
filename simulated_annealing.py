import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import matplotlib.pyplot as plt
import time


def create_image():
    return np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)


# cuda code
cuda_code = """
#include <cmath>
__device__ float lcg_random(unsigned int *seed)
{
    const unsigned int a = 1664525;
    const unsigned int c = 1013904223;
    *seed = a * (*seed) + c;
    return static_cast<float>(*seed) / static_cast<float>(UINT_MAX);
}

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


__global__ void simulatedAnnealing(unsigned char* inputImage, unsigned char* outputImage,
                                   float temperature, float coolingRate, int* swaps,
                                   int width, int height, unsigned int seed)
{

    extern __shared__ unsigned char sharedMem[];
    extern __shared__ int sharedEnergy[];

    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int blockWidth = gridDim.x * blockDimX;

    int globalIdx = threadIdx.x + blockIdx.x * blockDimX;
    int globalIdy = threadIdx.y + blockIdx.y * blockDimY;

    int localIdx = threadIdx.x;
    int localIdy = threadIdx.y;

    int imageIndex = (globalIdy * blockWidth + globalIdx) * 3;

    if (globalIdx < 32 && globalIdy < 32)
    {
        for (int i = 0; i < 3; i++)
        {
            sharedMem[imageIndex + i] = inputImage[imageIndex + i];
        }
    }

    __syncthreads();

    if (globalIdx < 32 && globalIdy < 32)
    {
        sharedEnergy[localIdy * blockDim.x + localIdx] = calculatePixelEnergy(sharedMem, imageIndex, 32);
    }

    __syncthreads();

    int startingEnergy = 0;
    if (localIdx == 0 && localIdy == 0)
    {
        for (int i = 0; i < blockDim.x * blockDim.y; ++i)
        {
            startingEnergy += sharedEnergy[i];
        }
    }

    __syncthreads();

    bool acceptSwap;

    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    if (localIdx < blockDimY)
    {
        int* swap = &swaps[localIdx * 4];

        int x1 = swap[0], y1 = swap[1];
        int x2 = swap[2], y2 = swap[3];

        int energyBefore = 0;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int checkX1 = x1 + dx, checkY1 = y1 + dy;
                int checkX2 = x2 + dx, checkY2 = y2 + dy;
                if (checkX1 >= 0 && checkX1 < width && checkY1 >= 0 && checkY1 < height)
                {
                    energyBefore += calculatePixelEnergy(sharedMem, (checkY1 * width + checkX1) * 3, width);
                }
                if (checkX2 >= 0 && checkX2 < width && checkY2 >= 0 && checkY2 < height && (dx != 0 || dy != 0))
                {
                    energyBefore += calculatePixelEnergy(sharedMem, (checkY2 * width + checkX2) * 3, width);
                }
            }
        }


        for (int c = 0; c < 3; c++)
        {
            unsigned char temp = sharedMem[(y1 * blockDim.x + x1) * 3 + c];
            sharedMem[(y1 * blockDim.x + x1) * 3 + c] = sharedMem[(y2 * blockDim.x + x2) * 3 + c];
            sharedMem[(y2 * blockDim.x + x2) * 3 + c] = temp;
        }

        int energyAfter = 0;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int checkX1 = x1 + dx, checkY1 = y1 + dy;
                int checkX2 = x2 + dx, checkY2 = y2 + dy;
                if (checkX1 >= 0 && checkX1 < width && checkY1 >= 0 && checkY1 < height)
                {
                    energyAfter += calculatePixelEnergy(sharedMem, (checkY1 * width + checkX1) * 3, width);
                }
                if (checkX2 >= 0 && checkX2 < width && checkY2 >= 0 && checkY2 < height && (dx != 0 || dy != 0))
                {
                    energyAfter += calculatePixelEnergy(sharedMem, (checkY2 * width + checkX2) * 3, width);
                }
            }
        }

        int dE = energyAfter - energyBefore;

        float probability = 1.0f;
        if (dE > 0)
        {
            probability = pow(2, -(float)dE / temperature);
        }

        unsigned int local_seed = seed + threadIdx.x + blockIdx.x * blockDim.x;
        float randomNum = lcg_random(&local_seed);

        acceptSwap = randomNum < probability;
    }

    __syncthreads();

    if (!acceptSwap)
    {
        //revert jer je vec promenjen za kalkulaciju energije
        for (int c = 0; c < 3; c++)
        {
            unsigned char temp = sharedMem[(y1 * width + x1) * 3 + c];
            sharedMem[(y1 * width + x1) * 3 + c] = sharedMem[(y2 * width + x2) * 3 + c];
            sharedMem[(y2 * width + x2) * 3 + c] = temp;
        }
    }

    __syncthreads();

    if (globalIdx < 32 && globalIdy < 32)
    {
        for (int i = 0; i < 3; i++)
        {
            outputImage[imageIndex + i] = sharedMem[imageIndex + i];
        }
    }

}


"""

# cuda sm
model = SourceModule(cuda_code)
simulated_annealing = model.get_function("simulatedAnnealing")

# mem aloc gpuarray umesto cuda.mem_alloc i cuda.memcpy valjda je ovako bolje
image = create_image()
image_gpu = gpuarray.to_gpu(image)
result_gpu = gpuarray.empty_like(image_gpu)

# sk parametri
temperature = np.float32(1000)
# promeni kao u zadatku kasnije cooling rate
cooling_rate = np.float32(0.003)

shared_memory_size = 3072 + 4096  # mozda treba da se menja

# block size x = 12 y = 1 vljd
block_size = (12, 8, 1)  # drugi br je broj pixela za promenu i kaluklaciju energije
grid_size = (1, 1)

# max iteracije
max_iterations = 10000

for iteration in range(max_iterations):
    total_swaps = block_size[1] * grid_size[1]

    dt = np.dtype([('x1', np.int32), ('y1', np.int32), ('x2', np.int32), ('y2', np.int32)], align=True)
    swaps = np.zeros((total_swaps,), dtype=dt)
    for i in range(total_swaps):
        x1, y1 = np.random.randint(0, 31, 2)
        direction = np.random.randint(0, 2)  # 0 desni, 1 donji
        x2, y2 = (x1 + direction, y1 + (1 - direction))

        while any((swaps[:i, :2] == [x1, y1]).all(axis=1)):
            x1, y1 = np.random.randint(0, 31, 2)
            x2, y2 = (x1 + direction, y1 + (1 - direction))

        swaps[i]['x1'], swaps[i]['y1'], swaps[i]['x2'], swaps[i]['y2'] = x1, y1, x2, y2

    swaps_gpu = gpuarray.to_gpu(swaps)
    seed = int(time.time())
    simulated_annealing(image_gpu, result_gpu, temperature, cooling_rate, swaps_gpu,
                        np.int32(32), np.int32(32), np.int32(seed), block=block_size, grid=grid_size,
                        shared=shared_memory_size)

    temperature *= (1 - cooling_rate)

new_image = result_gpu.get()

plt.imshow(new_image)
plt.show()
