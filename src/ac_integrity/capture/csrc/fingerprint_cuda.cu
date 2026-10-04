#include "blake3_core.h"

// Each block hashes up to 256 standard 1024-byte chunks, then combines their
// chaining values in order. Only the final tree root receives the ROOT flag.
// Shared memory holds 8 KiB, independent of tensor length. Inactive threads
// participate in every barrier; the final odd subtree is carried unchanged.
__device__ void reduce_subtree(unsigned int nodes[256][8], int count,
                                bool root, unsigned int* output) {
  const int tid = threadIdx.x;
  while (count > 1) {
    const int parents = (count + 1) / 2;
    unsigned int value[8];
    if (tid < parents) {
      if (2 * tid + 1 < count)
        b3_parent(nodes[2 * tid], nodes[2 * tid + 1],
                  root && count == 2, value);
      else
        for (int i = 0; i < 8; ++i) value[i] = nodes[2 * tid][i];
    }
    // Read all children before compacting parents over the same array.
    __syncthreads();
    if (tid < parents)
      for (int i = 0; i < 8; ++i) nodes[tid][i] = value[i];
    __syncthreads();
    count = parents;
  }
  if (tid == 0)
    for (int i = 0; i < 8; ++i) output[i] = nodes[0][i];
}

extern "C" __global__ void fingerprint_kernel(
    const unsigned char* bytes, unsigned long long byte_count,
    unsigned int* output) {
  __shared__ unsigned int nodes[256][8];
  unsigned long long chunks = (byte_count + 1023) / 1024;
  if (chunks == 0) chunks = 1;
  const unsigned long long start = static_cast<unsigned long long>(blockIdx.x) * 256;
  const unsigned long long index = start + threadIdx.x;
  const int active = chunks - start < 256 ? static_cast<int>(chunks - start) : 256;
  if (index < chunks) {
    const unsigned long long begin = index * 1024;
    const unsigned long long remaining = byte_count - begin;
    b3_chunk(bytes + begin, remaining < 1024 ? remaining : 1024,
             index, chunks == 1, nodes[threadIdx.x]);
  }
  __syncthreads();
  reduce_subtree(nodes, active, gridDim.x == 1, output + blockIdx.x * 8);
}

extern "C" __global__ void parent_fingerprint_kernel(
    const unsigned int* input, unsigned long long node_count,
    unsigned int* output) {
  __shared__ unsigned int nodes[256][8];
  const unsigned long long start = static_cast<unsigned long long>(blockIdx.x) * 256;
  const unsigned long long index = start + threadIdx.x;
  const int active = node_count - start < 256 ? static_cast<int>(node_count - start) : 256;
  if (index < node_count)
    for (int i = 0; i < 8; ++i) nodes[threadIdx.x][i] = input[index * 8 + i];
  __syncthreads();
  reduce_subtree(nodes, active, gridDim.x == 1, output + blockIdx.x * 8);
}

extern "C" __global__ void compare_fingerprint_kernel(
    const unsigned int* original, const unsigned int* recomputed,
    unsigned long long* mismatch, unsigned int structural_failure) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  bool different = structural_failure != 0;
  for (int word = 0; word < 8; ++word)
    different = different || original[word] != recomputed[word];
  mismatch[0] = different ? 1ULL : 0ULL;
}
