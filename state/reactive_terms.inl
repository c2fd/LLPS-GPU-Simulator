#pragma once

__device__ inline myreal motility_device(myreal phi){
	return 4.*pow2(phi*(1.-phi));
}

__device__ inline myreal init_q_device(myreal phi){
	return phi * (1.0 - phi);
}

__device__ inline myreal  partial_q_device(myreal phi){
	return 1.-2.*phi;
}

__device__ inline myreal ToComplex(const myreal phi, const myreal W, const myreal R){
	return NP.c1 * W * R; // * (1.-phi);
}
__device__ inline myreal FromComplex(const myreal phi, const myreal W, const myreal R){
	//return NP.c2 * phi * (1.-phi);
	if( phi > 0.5)
		return 0.;
	else
		return NP.c2 * phi;
}

__device__ inline myreal fphi(const myreal phi, const myreal W, const myreal R){
	return ToComplex(phi,W,R) - FromComplex(phi,W,R);
}

__device__ inline myreal fW(const myreal phi, const myreal W, const myreal R){
	return -ToComplex(phi,W,R) + FromComplex(phi,W,R);
}

__device__ inline myreal fR(const myreal phi, const myreal W, const myreal R){
	return -ToComplex(phi,W,R) + FromComplex(phi,W,R);
}

/*
   __global__ void partial_q_kernel(myreal *rhs, myreal *phi){
   let_idx_no_quit(idx,i,j,k);

   myreal ret = 0.;
   if(!is_pad(i,j,k)){
   ret = partial_q_device(phi[idx]);
   }
   rhs[idx] = ret;
   }
   */
