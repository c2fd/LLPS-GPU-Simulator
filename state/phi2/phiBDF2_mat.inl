#pragma once
#include <string>
//#include "phicortex_pre.inl"

namespace phiBDF2{

	__global__ void phiBDF2_motility(myreal *M, const myreal *phi){
		let_idx_no_quit(idx,i,j,k);

		myreal ret = 0.;
		if(!is_pad(i,j,k))
			ret = motility_device(phi[idx]);
		M[idx] = ret;
	}

	__global__ void phiBDF2_rhs1(myreal *tmp1){
		let_idx_no_quit(idx,i,j,k);

		myreal ret = 0.;
		if(!is_pad(i,j,k))
			ret += ST.parqq[idx] * ( - ST.parqq[idx] * (4./3.*ST.phi[idx]-1./3.*ST.phi_old[idx]) + (4./3.*ST.qq[idx]-1./3.*ST.qq_old[idx]));
		tmp1[idx] = ret;
	}


	__global__ void phiBDF2_rhs(myreal *rhs, const myreal *tmp1, const myreal *M){
		let_idx_no_quit(idx,i,j,k);

		myreal ret = 0.;
		if(!is_pad(i,j,k)){
			ret = ST.idt_2 * (4.* ST.phi[idx]- ST.phi_old[idx]);
			//ret+= NP.Lambda * laplacian(tmp1,idx);
			ret+= NP.Lambda * div_scalar_grad(M,tmp1,idx);
			ret+= fphi(ST.phi_bar[idx],ST.W_bar[idx],ST.R_bar[idx]);
		}
		rhs[idx] = ret;
	}


	__global__ void phiBDF2_mul1(myreal *tmp1, myreal *x){
		let_idx_no_quit(idx,i,j,k);

		myreal ret = 0.;
		if(!is_pad(i,j,k)){
			ret += pow2(ST.parqq[idx]) * x[idx];
			ret -= NP.Gamma_1 * laplacian(x,idx);
		}
		tmp1[idx] = ret;
	}

	__global__ void phiBDF2_mul(myreal *x, myreal *y, const myreal *tmp1, const myreal *M){
		let_idx_no_quit(idx,i,j,k);

		bool pad = is_pad(i,j,k);
		myreal ret = 0.;
		if(!pad)
			ret = ST.idt3_2 * x[idx] - NP.Lambda *div_scalar_grad(M,tmp1,idx);
			//ret = ST.idt3_2 * x[idx] - NP.Lambda *laplacian(tmp1,idx);
		y[idx] = ret;
		__syncthreads();

		if(pad)
			x[idx] = 0.;
	}
}




void PhiBDF2Mat::get_RHS(Vec &rhs) const {
	using namespace phiBDF2;

	Vec tmp1(gp.Ndata,0.);
	Vec M(gp.Ndata,0.);
	phiBDF2::phiBDF2_motility<<<gp.dimGrid,gp.dimBlock>>>(cast(M),cast(s->phi_bar)); pad_phi(M); check();
	phiBDF2::phiBDF2_rhs1<<<gp.dimGrid,gp.dimBlock>>>(cast(tmp1)); pad_phi(tmp1); check();
	phiBDF2::phiBDF2_rhs <<<gp.dimGrid,gp.dimBlock>>>(cast(rhs),cast(tmp1),cast(M)); check();
	check("rhs");
}


void PhiBDF2Mat::operator()(const Vec &x_, Vec &y) const {
	Vec &x = const_cast<Vec&>(x_);
	using namespace phiBDF2;

	pad_phi(x);

	Vec tmp1(gp.Ndata,0.);
	Vec M(gp.Ndata,0.);
	phiBDF2::phiBDF2_motility<<<gp.dimGrid,gp.dimBlock>>>(cast(M),cast(s->phi_bar)); pad_phi(M); check();
	phiBDF2::phiBDF2_mul1<<<gp.dimGrid,gp.dimBlock>>>(cast(tmp1),cast(x)); pad_phi(tmp1); check();
	phiBDF2::phiBDF2_mul<<<gp.dimGrid,gp.dimBlock>>>(cast(x),cast(y),cast(tmp1),cast(M));  check();
	check("operator");
}
