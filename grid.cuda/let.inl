#pragma once
#include "grid_params.h"



#define let_ijk(i,j,k)				\
    int j = blockIdx.y*blockDim.y+threadIdx.y;\
    int k = blockIdx.x*blockDim.x+threadIdx.x; \
    int i = NULL; \

#define let_idx_no_quit(idx,i,j,k)		\
    let_ijk(i,j,k);				\
    int idx = j*GP.sy + k;

#define let_quit(i,j,k)					\
    if ( j>=GP.Nyp || k>=GP.Nzp)  return;

#define let_idx(idx,i,j,k)			\
    let_idx_no_quit(idx,i,j,k);			\
    let_quit(i,j,k)

#define let_idx_no_pad(idx,i,j,k)		\
    let_idx_no_quit(idx,i,j,k);                 \
    if (is_pad(i,j,k)) return;


/*
#define let_idx_shared(idx,sidx,i,j,k)			\
    let_ijk(i,j,k);					\
    int idx = j*GP.sy + k;			\
    int sidx = (threadIdx.z*blockDim.y + threadIdx.y)	\
	* blockDim.x + threadIdx.x;
*/
// Note:
//    It's caller responsibility to call  let_quit(i,j,k);
//    We don't do this since some callers might need sidx=0,1,2 to exist.
// Example:
//     if (sidx<3)
//         u[sidx] = velo + sidx*GP.Ndata


#define ijk_(d)      ((d)==0 ? i : ((d)==1 ? j : k))


// Use the template version for type safety.
//
// #define pick3(a,b,c,d)                          \
//     ((d)==0 ? (a) : ((d)==1 ? (b) : (c)))
//
template <typename T>
T pick3(T b, T c, int d) {
    return ((d)==0 ? (b) : (c));
}


//================

//
// Restrict the domain (idx) to avoid page fault.
//
#define let_idx_1d_(idx, code_block, idx_end)	     \
    int idx = blockDim.x * blockIdx.x + threadIdx.x; \
    const int inc_ = gridDim.x * blockDim.x;	     \
    while(idx < idx_end) {			     \
	code_block;				     \
	idx += inc_;				     \
    }

// To prevent seg-fault, we need idx + GP.sx < GP.Ndata
#define let_idx_1d(idx, code_block)			\
    let_idx_1d_(idx, code_block, GP.Ndata-2*GP.sy)

#define LAUNCH_CONFIGURATION_1D \
    <<<ceil_div(gp.Ndata-2*gp.sy,gp.BLOCK_SIZE_1D*gp.BLOCK_SIZE_1D_MULT), gp.BLOCK_SIZE_1D>>>

#define let_idx_1d_with_pad(idx, code_block)	\
    let_idx_1d_(idx, code_block, GP.Ndata)

#define LAUNCH_CONFIGURATION_1D_WITH_PAD				\
    <<<ceil_div(gp.Ndata,gp.BLOCK_SIZE_1D*gp.BLOCK_SIZE_1D_MULT), gp.BLOCK_SIZE_1D>>>





//================


__host__ __device__
inline myreal pow2(myreal x) {return x*x;}
__host__ __device__
inline myreal pow3(myreal x) {return x*x*x;}
__host__ __device__
inline myreal pow4(myreal x) {return x*x*x*x;}

__device__ myreal ave4(const myreal *v, int idx, int s1, int s2) {
    return myreal(0.25) * (
	v[idx] + v[idx+s1] + v[idx+s2] + v[idx+s1+s2]);
}
