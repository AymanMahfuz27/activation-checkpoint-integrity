__device__ __forceinline__ unsigned int mix32(unsigned int value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__device__ __forceinline__ unsigned int rotate_left32(
    unsigned int value, int bits) {
  return (value << bits) | (value >> (32 - bits));
}

extern "C" __global__ void fingerprint_kernel(
    const unsigned char* bytes,
    unsigned long long byte_count,
    unsigned int* output) {
  const int threads = 256;
  const int warp_size = 32;
  const int warps_per_block = threads / warp_size;

  __shared__ unsigned int warp_sums[4][warps_per_block];

  const unsigned int position_seeds[4] = {
      0x243f6a89U, 0xa4093823U, 0x13198a2fU, 0x082efa99U};
  const unsigned int length_seeds_low[4] = {
      0x452821e7U, 0x38d01377U, 0xbe5466cfU, 0x34e90c6dU};
  const unsigned int length_seeds_high[4] = {
      0xc0ac29b7U, 0xc97c50ddU, 0x3f84d5b5U, 0xb5470917U};

  const unsigned long long word_count = (byte_count + 3) / 4;
  const unsigned long long global_thread =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long grid_stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;

  unsigned int sums[4] = {0, 0, 0, 0};

  for (unsigned long long word_index = global_thread;
       word_index < word_count;
       word_index += grid_stride) {
    const unsigned long long begin = word_index * 4;
    unsigned int word = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      const unsigned long long position = begin + byte;
      if (position < byte_count) {
        word |= static_cast<unsigned int>(bytes[position]) << (8 * byte);
      }
    }
    const unsigned int position = static_cast<unsigned int>(word_index);
    const unsigned int first = mix32(position ^ position_seeds[0]);
    const unsigned int second = mix32(position ^ position_seeds[1]);
    sums[0] += word * (first | 1U);
    sums[1] += word * ((rotate_left32(first, 11) ^ position_seeds[2]) | 1U);
    sums[2] += word * (second | 1U);
    sums[3] += word * ((rotate_left32(second, 17) ^ position_seeds[3]) | 1U);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned int low = static_cast<unsigned int>(byte_count);
    const unsigned int high = static_cast<unsigned int>(byte_count >> 32);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      sums[lane] += (low + 1U) * length_seeds_low[lane];
      sums[lane] += (high + 1U) * length_seeds_high[lane];
    }
  }

  const int warp = threadIdx.x / warp_size;
  const int lane_in_warp = threadIdx.x % warp_size;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      sums[lane] += __shfl_down_sync(0xffffffffU, sums[lane], offset);
    }
    if (lane_in_warp == 0) {
      warp_sums[lane][warp] = sums[lane];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      unsigned int block_sum =
          lane_in_warp < warps_per_block ? warp_sums[lane][lane_in_warp] : 0;
#pragma unroll
      for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffffU, block_sum, offset);
      }
      if (lane_in_warp == 0) {
        atomicAdd(output + lane, block_sum);
      }
    }
  }
}

extern "C" __global__ void compare_fingerprint_kernel(
    const unsigned int* original,
    const unsigned int* recomputed,
    unsigned long long* mismatch,
    unsigned int structural_failure) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  bool different = structural_failure != 0;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    different = different || original[lane] != recomputed[lane];
  }
  mismatch[0] = different ? 1ULL : 0ULL;
}
