#pragma once

__global__ void RMat_rhs(myreal *rhs){
	let_idx_no_quit(idx,i,j,k);

	myreal ret = 0.;

	if(!is_pad(i,j,k))
		ret = ST.idt_2 *(4.*ST.R[idx]-ST.R_old[idx]) + fR(ST.phi_bar[idx],ST.W_bar[idx],ST.R_bar[idx]);
	rhs[idx] = ret;
}

__global__ void RMat_mul(myreal *x, myreal *y){
	let_idx_no_quit(idx,i,j,k);

	myreal ret = 0.;

	bool pad = is_pad(i,j,k);
	if(!pad)
		ret = ST.idt3_2 * x[idx] - NP.LambdaR * laplacian(x,idx);
	y[idx] = ret;
	__syncthreads();
	if(pad)
		x[idx] = 0.;
}

void RMat::get_RHS(Vec &rhs) const {
	RMat_rhs<<<gp.dimGrid,gp.dimBlock>>>(cast(rhs));
	check("RMat::get_RHS");
}

void RMat::operator()(const Vec &x_, Vec &y) const {
	Vec &x = const_cast<Vec&>(x_);
	pad_reflex(x);

	RMat_mul<<<gp.dimGrid,gp.dimBlock>>>(cast(x),cast(y));
	check("RMat::mul");
}
