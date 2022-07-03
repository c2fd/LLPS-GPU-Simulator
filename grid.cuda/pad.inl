#pragma once
#include "tools.h"
#include "grid.cuda/pad.h"
#include "grid.cuda/pad_device.inl"



__global__ void pad_reflex(myreal *v, int direction = (1<<Ndim)-1) {
	let_idx_no_quit(idx,i,j,k);
	pad_reflex(v,idx,i,j,k,direction);
}

__global__ void pad_phi(myreal *v) {
	let_idx_no_quit(idx,i,j,k);
	pad_phi(v,idx,i,j,k);
}

__global__ void pad_zeros(myreal *v) {
	let_idx_no_quit(idx,i,j,k);
	pad_zeros(v,idx,i,j,k);
}

void pad_reflex(Vec &v) {
	assert(v.size()%gp.Ndata == 0);
	for (int ii=0; ii<v.size(); ii+=gp.Ndata)
		pad_reflex <<<gp.dimGrid,gp.dimBlock>>> (cast(v) + ii);
}

void pad_phi(Vec &v) {
	pad_phi <<<gp.dimGrid,gp.dimBlock>>> (cast(v));
}

void pad_zeros(Vec &v){
	pad_zeros<<<gp.dimGrid,gp.dimBlock>>>(cast(v));
}

__global__ void padpad_vec(const myreal *src, myreal *target, int action, int sdata, int sy, int n) {
	let_idx(idx,i,j,k);
	if (abs(action)==2) {
		if (is_pad(i,j,k))
			return;
		j -= Npad;
		k -= Npad;
	}
	int id = j*sy + k;
	for (int ii=0; ii<n; ii++) {
		if (action>0)
			target[idx+ii*GP.Ndata] = src[id+ii*sdata];
		else
			target[id+ii*sdata] = src[idx+ii*GP.Ndata];
	}
}

void padpad_vec(const myreal *src, myreal *target, int action, int n) {
	assert(abs(action)==1 || abs(action)==2);

	int sdata, sy;
	if (abs(action)==1) {
		sy = gp.Nzp;
		sdata = gp.Nyp * sy;
	} else {
		sy = gp.Nz;
		sdata = gp.Ny * sy;
	}

	padpad_vec <<<gp.dimGrid,gp.dimBlock>>> (src, target, action, sdata, sy, n);
}
