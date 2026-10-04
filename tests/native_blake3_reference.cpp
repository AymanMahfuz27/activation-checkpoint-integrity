// Execute the CUDA compression primitives on CPU, independently of the Python
// library. The traversal mirrors the 256-chunk CUDA subtree grouping, including
// odd tails and the ROOT flag. GPU synchronization is tested separately.
#include <algorithm>
#include <array>
#include <vector>
#include "blake3_core.h"

using Cv = std::array<unsigned int, 8>;

static Cv reduce(std::vector<Cv> nodes, bool root) {
  while (nodes.size() > 1) {
    std::vector<Cv> parents;
    for (size_t i = 0; i < nodes.size(); i += 2) {
      Cv cv;
      if (i + 1 < nodes.size())
        b3_parent(nodes[i].data(), nodes[i + 1].data(),
                  root && nodes.size() == 2, cv.data());
      else
        cv = nodes[i];
      parents.push_back(cv);
    }
    nodes = parents;
  }
  return nodes[0];
}

extern "C" void aci_blake3_cpu(const unsigned char* bytes,
                                unsigned long long size, unsigned char* out) {
  unsigned long long chunks = std::max(1ULL, (size + 1023) / 1024);
  unsigned long long blocks = (chunks + 255) / 256;
  std::vector<Cv> level;
  for (unsigned long long block = 0; block < blocks; ++block) {
    std::vector<Cv> nodes;
    for (unsigned long long i = block * 256; i < std::min(chunks, (block + 1) * 256); ++i) {
      Cv cv;
      auto begin = i * 1024;
      b3_chunk(bytes + begin, std::min(1024ULL, size - begin),
               i, chunks == 1, cv.data());
      nodes.push_back(cv);
    }
    level.push_back(reduce(nodes, blocks == 1));
  }
  while (level.size() > 1) {
    std::vector<Cv> next;
    for (size_t begin = 0; begin < level.size(); begin += 256) {
      auto end = std::min(level.size(), begin + 256);
      std::vector<Cv> group(level.begin() + begin, level.begin() + end);
      next.push_back(reduce(group, level.size() <= 256));
    }
    level = next;
  }
  for (int word = 0; word < 8; ++word)
    for (int byte = 0; byte < 4; ++byte)
      out[word * 4 + byte] = static_cast<unsigned char>(level[0][word] >> (byte * 8));
}
