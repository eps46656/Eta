#ifndef ETA_DEFINE_CUH
#define ETA_DEFINE_CUH

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <array>
#include <limits>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define ETA_offset(type, member) (size_t(&((type*)(0))->member))
#define ETA_get_type_with_member(type, member, member_inst)                    \
    (reinterpret_cast<type*>(ETA_ptr_addr(member_inst) -                       \
                             ETA_offset(type, member)))

#define ETA_inf (1e32f * 1e32f * 1e32f * 1e32f * 1e32f * 1e32f * 1e32f * 1e32f)
#define ETA_pi (3.14159265358979323f)

#define ETA_cpu (0)
#define ETA_gpu (1)

#define ETA_deg (ETA_pi / 180.0f)

#define ETA_max_pixel_index (1920 * 1080 * 16)

#define TID (threadIdx.x)
#define GLOBAL_TID (blockIdx.x * blockDim.x + threadIdx.x)

#define ETA_MUL (8192.0f)

#define ETA_BRANCH_NUMBER (1)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define ETA_print_pos                                                          \
    { ::printf("pos: %s:%d\n", __FILE__, __LINE__); }

#define ETA_interrupt                                                          \
    {                                                                          \
        ETA_print_pos;                                                         \
        printf("interrupting... ");                                            \
        char c;                                                                \
        scanf("%c", &c);                                                       \
    }

#define ETA_exit                                                               \
    {                                                                          \
        ::printf("\n");                                                        \
        assert(false);                                                         \
    }

#define ETA_throw(desc)                                                        \
    {                                                                          \
        ::printf("throw at %s:%d\n", __FILE__, __LINE__);                      \
        ::printf(desc);                                                        \
        ETA_exit;                                                              \
    }

#define ETA_debug_flag 1

#if ETA_debug_flag
    #include <assert.h>

    #define ETA_debug
    #define ETA_debug_if(x) if (x)

    #define ETA_debug_throw(desc)                                              \
        {                                                                      \
            ::printf("throw at %s:%d\n", __FILE__, __LINE__);                  \
            ::printf(desc);                                                    \
            ETA_exit;                                                          \
        }
    #define ETA_debug_assert(cond, desc)                                       \
        if (!(cond)) { ETA_debug_throw(desc); }

    #define ETA_CheckCudaError(e)                                              \
        {                                                                      \
            cudaError_t error{ e };                                            \
            ETA_debug_if(error != cudaSuccess) {                               \
                ::printf("%s", cudaGetErrorString(error));                     \
                ETA_exit;                                                      \
            }                                                                  \
        }
    #define ETA_CheckLastCudaError                                             \
        {                                                                      \
            ETA_print_pos;                                                     \
            cudaDeviceSynchronize();                                           \
            ETA_CheckCudaError(cudaGetLastError())                             \
        }

#else
    #define ETA_debug if constexpr (false)
    #define ETA_debug_if(x) if constexpr (false)

    #define ETA_debug_throw(desc)
    #define ETA_debug_assert(cond, desc)

    #define ETA_CheckCudaError(e)
    #define ETA_CheckLastCudaError
#endif

#endif