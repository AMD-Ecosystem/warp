Declare the tile helpers' shared-memory arrays as raw aligned storage on CUDA as well as HIP, so NVRTC no longer diagnoses dynamic initialization of shared vector arrays in tile reductions.
