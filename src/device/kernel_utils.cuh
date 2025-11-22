#pragma once
#include <cuda_runtime.h>

namespace csrspmm::kernel {


__device__ __forceinline__ int lane_id()
{
    int id;
    asm("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}





















} // namespace csrspmm::kernel
