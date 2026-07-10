#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_tile.h"

namespace ct = cuda::tiles;
using namespace ct::literals;

// #include <cmath>
// The size of the grid is 64 x 64
const int N = 64;
//constexpr int CHUNK = 16;
constexpr int CHUNK = 1;
// An auxiliary function to calculate the euclidean distance between 2 points
__tile__ int distance(int x1,int y1,int x2,int y2) {
  int aux_x = x2 - x1;
  int aux_y = y2 - y1;
  aux_x = aux_x * aux_x;
  aux_y = aux_y * aux_y;
  int sum = aux_x + aux_y;
  // return (int) ct::sqrt(sum);
  return sum;
  }

// The kernel that calculates the approximate Voronoi Diagram
__tile_global__ void calcVoronoi(int* __restrict__ data,
                // size_t pitch,
                int width, int height,
                int nSeeds,int2 __restrict__ *seeds)
{
        /*
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
        */

    data = ct::assume_aligned(data, 64_ic);
    //int x = ct::bid().x;
    //int y = ct::bid().y;
    auto [x,y,z] = ct::bid();
    // Attach a shape to the raw pointer
    auto data_span = ct::tensor_span{data, ct::extents{64_ic,64_ic}};
    // Partition the span into a tile space of fixed size tiles
    auto view_data = ct::partition_view{data_span,ct::shape<CHUNK,CHUNK>{}};
    // load the tile from the input partition
    std::size_t num_k = N / CHUNK; // (K + tk - 1) / tk;
    //for (auto k : ct::irange(std::size_t{0}, num_k)) {
        auto tile_data = view_data.load_masked(x,y);
    /*
    if (x < width && y < height) {
        int* row = reinterpret_cast<int*>(reinterpret_cast<char*>(data) + y * pitch);
        */
        // Calculate the id of the closest seed
        int closestSeed = -1;
        int closestDistance = N*N;
        for(int i = 0;i < nSeeds;i++) {
                int seedx = seeds[i].x;
                int seedy = seeds[i].y;
                int dist = distance(seedx,seedy,(int)x,(int)y);
                if (dist < closestDistance) {
                   closestDistance = dist;
                   closestSeed = i;
                }
        }
    //}
        tile_data = tile_data + closestSeed;
        view_data.store_masked(tile_data,x,y);
    // }
}

int main()
{
    int width = N;
    int height = N;
    size_t pitch = 0;
    int* d_data = nullptr;
    // For a simple example, let's hardcode the number of seeds
    // and the values of the seeds
    int nSeeds = 4;
    int2 seeds[nSeeds];
    seeds[0] = make_int2(0,0);
    seeds[1] = make_int2(N,0);
    seeds[2] = make_int2(N,N);
    seeds[3] = make_int2(0,N);
    std::printf("%d %d\n",seeds[2].x,seeds[2].y);
    // Now allocate an array in the device memory for the seeds
    int2 *deviceSeeds = nullptr;
    cudaError_t err = cudaMalloc((void **)&deviceSeeds,nSeeds*sizeof(int2));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;

        }
// Copy the seeds from the host to device memory
    err = cudaMemcpy(deviceSeeds, seeds,nSeeds * sizeof(int2),
    cudaMemcpyHostToDevice);

    if (err != cudaSuccess) { 
        std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    // Allocate the 2d matrix that will contain the approximate voronoi diagram
//    err = cudaMallocPitch(&d_data, &pitch, width * sizeof(int), height);
    err = cudaMalloc((void **)&d_data, width * sizeof(int) * height);
    if (err != cudaSuccess) {
        // std::fprintf(stderr, "cudaMallocPitch failed: %s\n", cudaGetErrorString(err));
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;

}


// Execute the kernel
    // dim3 block(16, 16);
    // dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // dim3 grid(N/CHUNK,N/CHUNK);
    dim3 grid(N,N);
    // calcVoronoi<<<grid, block>>>(d_data, pitch, width, height,nSeeds,deviceSeeds);
    calcVoronoi<<<grid>>>(d_data, // pitch,
                    width, height,nSeeds,deviceSeeds);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }
// Copy the 2D array from the device to the host memory
    int* h_data = new int[width * height];
    // err = cudaMemcpy2D(h_data, width * sizeof(int), d_data, pitch, width * sizeof(int), height, cudaMemcpyDeviceToHost);
    err = cudaMemcpy(h_data, d_data, height* width * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy2D failed: %s\n", cudaGetErrorString(err));
        delete[] h_data;
        cudaFree(d_data);
        return 1;
    }
// Print sample data
//    std::printf("sample [0][0] = %d, [1023][1023] = %d\n", h_data[0], h_data[1023 * width + 1023]);
    for(int i = 0;i < N;i++) {
      for(int j = 0;j < N;j++) {
        std::printf("%d", h_data[j*N+i]);
        }
        std::printf("\n");
        }
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(deviceSeeds);
    return 0;
}