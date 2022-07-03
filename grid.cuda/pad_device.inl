#pragma once
#include "grid.cuda/grid_params.inl"
#include "grid.cuda/gallery.inl"


inline __device__ bool is_padpad(int i, int j, int k) {
	return (j>=GP.Nyp || k>=GP.Nzp);
}

inline __device__ bool is_pad(int i, int j, int k) {
	return ( (j < Npad) || (j >= GP.Ny1) ||(k < Npad) || (k >= GP.Nz1));
}

inline __device__ int wrap_idx(int idx, int i, int dir) {
	bool periodic = GP.periodicity & (1<<dir);
	int ny1 = (&GP.Ny1)[dir];
	if (periodic) {
		int wrap = (&GP.wrap_y)[dir];
		if (i < Npad)  return idx + wrap;
		if (ny1 <= i)  return idx - wrap;
	} else {
		int sy = (&GP.sy)[dir];
		if (i < Npad)  return idx + sy * (2*(Npad-i)-1);
		if (ny1 <= i)  return idx - sy * (2*(i-ny1)+1);
	}
	return idx;
}

inline __device__ void pad_reflex(myreal *v, int idx, int i, int j, int k, int direction = (1<<Ndim)-1) {

	if (is_padpad(i,j,k)) {
		v[idx] = myreal(0);
		return;
	}

	int id = idx;
	if (direction & 1)  id = wrap_idx(id,j,0);
	if (direction & 2)  id = wrap_idx(id,k,1);
	if (id!=idx)
		v[idx] = v[id];
}


inline __device__ void pad_phi(myreal *v, int idx, int i, int j, int k) {
	pad_reflex(v, idx,i,j,k);
}

inline __device__ void pad_zeros(myreal *v, int idx, int i, int j, int k) {
	if (is_pad(i,j,k))
		v[idx] = myreal(0);
}
