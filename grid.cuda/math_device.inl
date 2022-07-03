#pragma once
#include "grid.cuda/grid_params.inl"


// inline __host__ __device__
// myreal clip(myreal *dest, myreal a, myreal b, int i) {
//     return min(max(dest[i],a),b);
// }

inline __device__
myreal laplacian(const myreal *v, int i) {
    return
    	GP._dz2 * (v[i+1]    -2*v[i]+v[i-1]    ) +
    	GP._dy2 * (v[i+GP.sy]-2*v[i]+v[i-GP.sy]);
}

inline __device__
myreal divergent(const myreal *v, const myreal *w, int i) {
    return
	GP._dz * (w[i] - w[i-GP.sz]) +
	GP._dy * (v[i] - v[i-GP.sy]);
}
inline __device__
myreal divergent(const myreal* velo, int i) {
    return divergent(velo, velo+GP.Ndata, i);
}



//================================================================


inline __device__
myreal div_scalar_velo(const myreal *a, const myreal *velo) {
    const myreal *v = velo;
    const myreal *w = v+GP.Ndata;
    return myreal(.5) * (
        + GP._dy * ((a[0] + a[GP.sy])*v[0] - (a[0] + a[-GP.sy])*v[-GP.sy])
        + GP._dz * ((a[0] + a[GP.sz])*w[0] - (a[0] + a[-GP.sz])*w[-GP.sz])
        );
}

inline __device__
myreal div_scalar_velo(const myreal *a, const myreal *velo, int idx) {
    return div_scalar_velo(a+idx,velo+idx);
}


inline __device__
myreal div_scalar_grad(const myreal *a, const myreal *b) {
    // return div (a grad b),  where a and b are both cell centered.
    return myreal(.5) * (
        + GP._dy2 * (+(a[ GP.sy] + a[0]) * (b[GP.sy] - b[0])
                     -(a[-GP.sy] + a[0]) * (b[0] - b[-GP.sy]))
        + GP._dz2 * (+(a[ GP.sz] + a[0]) * (b[GP.sz] - b[0])
                     -(a[-GP.sz] + a[0]) * (b[0] - b[-GP.sz]))
        );
}

inline __device__
myreal div_scalar_grad(const myreal *a, const myreal *b, int idx) {
    return div_scalar_grad(a+idx,b+idx);
}



inline __device__
myreal v_dot_grad(const myreal *velo, const myreal *x) {
    const myreal *v = velo;
    const myreal *w = v+GP.Ndata;

    return
    	+ (.5*GP._dy) * (v[0]*(x[GP.sy]+x[0]) - v[-GP.sy]*(x[0]+x[-GP.sy]))
    	+ (.5*GP._dz) * (w[0]*(x[GP.sz]+x[0]) - w[-GP.sz]*(x[0]+x[-GP.sz]));

}
inline __device__
myreal v_dot_grad(const myreal *velo, const myreal *x, int idx) {
    return v_dot_grad(velo+idx, x+idx);
}



inline __device__
myreal grad_squared(const myreal *phi, int idx) {
    return 
	+ pow2(.5*GP._dy*(phi[idx+GP.sy] - phi[idx-GP.sy]))
	+ pow2(.5*GP._dz*(phi[idx+GP.sz] - phi[idx-GP.sz]));
}








//================================================================


//
// Reduce add instructions by folding idx into the pointer
//

inline __device__
myreal laplacian(myreal *v) {
    return
    	GP._dz2 * (v[1]    -2*v[0]+v[-1]    ) +
    	GP._dy2 * (v[GP.sy]-2*v[0]+v[-GP.sy]);
}
