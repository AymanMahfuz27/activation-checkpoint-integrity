// Standard unkeyed BLAKE3 compression, shared by CUDA and native CPU tests.
// Algorithm: https://github.com/BLAKE3-team/BLAKE3-specs
// This implements the first 32 output bytes, without keyed hashing or XOF.
#ifndef ACI_BLAKE3_CORE_H
#define ACI_BLAKE3_CORE_H

#ifdef __CUDACC__
#define ACI_INLINE __device__ __forceinline__
#else
#define ACI_INLINE inline
#endif

ACI_INLINE unsigned int b3_rotate(unsigned int x, int n) {
  return (x >> n) | (x << (32 - n));
}

ACI_INLINE void b3_iv(unsigned int* cv) {
  const unsigned int initial[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  #pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = initial[i];
}

ACI_INLINE void b3_g(unsigned int* s, int a, int b, int c, int d,
                     unsigned int x, unsigned int y) {
  s[a] += s[b] + x;
  s[d] = b3_rotate(s[d] ^ s[a], 16);
  s[c] += s[d];
  s[b] = b3_rotate(s[b] ^ s[c], 12);
  s[a] += s[b] + y;
  s[d] = b3_rotate(s[d] ^ s[a], 8);
  s[c] += s[d];
  s[b] = b3_rotate(s[b] ^ s[c], 7);
}

ACI_INLINE void b3_compress(const unsigned int* cv,
                            const unsigned int* message,
                            unsigned long long counter,
                            unsigned int block_bytes, unsigned int flags,
                            unsigned int* output) {
  // Inputs describe one 64-byte compression block. Output is its first eight
  // words, used either as a chaining value or as the ROOT digest.
  unsigned int s[16], m[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = cv[i];
  s[8] = 0x6a09e667U; s[9] = 0xbb67ae85U;
  s[10] = 0x3c6ef372U; s[11] = 0xa54ff53aU;
  s[12] = static_cast<unsigned int>(counter);
  s[13] = static_cast<unsigned int>(counter >> 32);
  s[14] = block_bytes; s[15] = flags;
  #pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = message[i];
  const int permutation[16] = {2, 6, 3, 10, 7, 0, 4, 13,
                               1, 11, 12, 5, 9, 14, 15, 8};
  #pragma unroll
  for (int round = 0; round < 7; ++round) {
    b3_g(s, 0, 4, 8, 12, m[0], m[1]);
    b3_g(s, 1, 5, 9, 13, m[2], m[3]);
    b3_g(s, 2, 6, 10, 14, m[4], m[5]);
    b3_g(s, 3, 7, 11, 15, m[6], m[7]);
    b3_g(s, 0, 5, 10, 15, m[8], m[9]);
    b3_g(s, 1, 6, 11, 12, m[10], m[11]);
    b3_g(s, 2, 7, 8, 13, m[12], m[13]);
    b3_g(s, 3, 4, 9, 14, m[14], m[15]);
    if (round != 6) {
      unsigned int next[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) next[i] = m[permutation[i]];
      #pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = next[i];
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) output[i] = s[i] ^ s[i + 8];
}

ACI_INLINE void b3_chunk(const unsigned char* bytes,
                         unsigned long long byte_count,
                         unsigned long long chunk_index, bool root,
                         unsigned int* output) {
  // Hash one <=1024-byte chunk. The last compression includes CHUNK_END;
  // a one-chunk message also includes ROOT and uses output-block counter zero.
  unsigned int cv[8];
  b3_iv(cv);
  unsigned long long blocks = (byte_count + 63) / 64;
  if (blocks == 0) blocks = 1;
  for (unsigned long long block = 0; block < blocks; ++block) {
    unsigned int m[16] = {};
    unsigned long long remaining = byte_count - block * 64;
    unsigned int count = remaining > 64 ? 64 : static_cast<unsigned int>(remaining);
    #pragma unroll
    for (int word = 0; word < 16; ++word) {
      #pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        unsigned int i = word * 4 + byte;
        if (i < count)
          m[word] |= static_cast<unsigned int>(bytes[block * 64 + i]) << (8 * byte);
      }
    }
    bool last = block + 1 == blocks;
    unsigned int flags = (block == 0 ? 1U : 0U) | (last ? 2U : 0U);
    if (root && last) flags |= 8U;
    b3_compress(cv, m, root && last ? 0 : chunk_index, count, flags, cv);
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) output[i] = cv[i];
}

ACI_INLINE void b3_parent(const unsigned int* left, const unsigned int* right,
                          bool root, unsigned int* output) {
  unsigned int cv[8], m[16];
  b3_iv(cv);
  #pragma unroll
  for (int i = 0; i < 8; ++i) { m[i] = left[i]; m[i + 8] = right[i]; }
  b3_compress(cv, m, 0, 64, 4U | (root ? 8U : 0U), output);
}

#endif
