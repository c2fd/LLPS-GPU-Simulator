#pragma once
#include "grid.cuda/math.h"
#include "grid.cuda/math_device.inl"


// __global__
// void clip(myreal *dest, myreal min, myreal max) {
//     let_idx_1d_with_pad(idx, 
// 			dest[idx] = clip(dest,min,max,idx));
// }
__global__
void laplacian(myreal *dest, const myreal *src) {
    let_idx_1d(idx, dest[idx] = laplacian(src,idx));
}
__global__
void divergent(myreal *dest, const myreal* velo) {
    let_idx_1d(idx, dest[idx] = divergent(velo,idx));
}
__global__
void div_scalar_velo(myreal *dest, const myreal *a, const myreal *velo) {
    let_idx_1d(idx, dest[idx] = div_scalar_velo(a,velo,idx));
}
__global__
void div_scalar_grad(myreal *dest, const myreal *a, const myreal *b) {
    let_idx_1d(idx, dest[idx] = div_scalar_grad(a,b,idx));
}
// __global__
// void add_div_scalar_grad(myreal *dest, const myreal *phi, const myreal *chem) {
//     let_idx_1d(idx,
// 	       dest[idx] += div_scalar_grad(phi,chem,idx));
// }
__global__
void v_dot_grad(myreal *dest, const myreal *velo, const myreal *x) {
    let_idx_1d(idx, dest[idx] = v_dot_grad(velo,x,idx));
}
__global__
void grad_squared(myreal *dest, const myreal *phi) {
    let_idx_1d(idx, dest[idx] = grad_squared(phi,idx));
}



//================

// void clip(Vec &dest, myreal min, myreal max) {
//     clip LAUNCH_CONFIGURATION_1D_WITH_PAD
// 	(cast(dest), min, max);
//     check();
// }

void laplacian(Vec &dest, const Vec &src, int i) {
    laplacian LAUNCH_CONFIGURATION_1D
	(cast(dest) + i*gp.Ndata + gp.sy,
	 cast(src)  + i*gp.Ndata + gp.sy);
    check();
}
void divergent(Vec &dest, const Vec3 &velo) {
    divergent LAUNCH_CONFIGURATION_1D
	(cast(dest)+gp.sy, cast(velo)+gp.sy);
    check();
}
void div_scalar_velo(Vec &dest, const Vec &a, const Vec3 &velo) {
    div_scalar_velo LAUNCH_CONFIGURATION_1D
	(cast(dest)+gp.sy, cast(a)+gp.sy, cast(velo)+gp.sy);
    check();
}
void div_scalar_grad(Vec &dest, const Vec &a, const Vec &b) {
    div_scalar_grad LAUNCH_CONFIGURATION_1D
	(cast(dest)+gp.sy, cast(a)+gp.sy, cast(b)+gp.sy);
    check();
}
// void add_div_scalar_grad(myreal *dest, const Vec &phi, const Vec &chem) {
//     add_div_scalar_grad LAUNCH_CONFIGURATION_1D
// 	(dest+gp.sx, cast(phi)+gp.sx, cast(chem)+gp.sx);
//     check();
// }
// void add_div_scalar_grad_(myreal *dest, const myreal *phi, const myreal *chem) {
//     add_div_scalar_grad LAUNCH_CONFIGURATION_1D
// 	(dest+gp.sx, phi+gp.sx, chem+gp.sx);
//     check();
// }
void v_dot_grad(Vec &dest, const Vec3 &velo, const Vec &x) {
    int n = dest.size(); 
    assert(n==x.size() && n%gp.Ndata == 0);
    for (int ii=0; ii<n; ii+=gp.Ndata) {
	v_dot_grad LAUNCH_CONFIGURATION_1D
	    (cast(dest) + ii + gp.sy,
	     cast(velo) + gp.sy, // sic
	     cast(x) + ii + gp.sy);
	check("v_dot_grad", dest);
    }
}
void grad_squared(Vec &dest, const Vec &phi) {
    int n = dest.size(); 
    assert(n==phi.size() && n%gp.Ndata == 0);
    for (int ii=0; ii<n; ii+=gp.Ndata) {
	grad_squared LAUNCH_CONFIGURATION_1D (cast(dest) + ii + gp.sy, cast(phi) + ii + gp.sy);
	check();
    }
}


// ================



__device__
inline int idx_plus_sxd(int idx, int i, int j, int k, int d) {
    int ii = ijk_(d);
    int sy     = (&GP.sy)[d];
    int Nyp    = (&GP.Nyp)[d];
    int wrap_y = (&GP.wrap_y)[d];

    int ret = idx+sy;
    if (ii==Nyp-1) {
        if (GP.periodicity & (1<<d)) // is periodic
            ret -= wrap_y;
        else
            ret -= (2*Npad+1)*sy;
    }
    return ret;
}

__global__
void subtract_grad(myreal *velo, const myreal *x) {
    let_idx(idx,i,j,k);
    for (int d=0; d<Ndim; d++) {
        myreal _dy = (&GP._dy)[d];
        int idx2 = idx_plus_sxd(idx,i,j,k,d);
        velo[d*GP.Ndata+idx] -= _dy * (x[idx2] - x[idx]);
    }
}
void subtract_grad(Vec3 &velo, const Vec &x) {
    // This function assumes that
    // - x's boundaries are either periodic or zero Neumann.
    // - Both velo and x are padded.
    // This is crucial!
    subtract_grad <<<gp.dimGrid,gp.dimBlock>>>
	(cast(velo), cast(x));
    check();
}
