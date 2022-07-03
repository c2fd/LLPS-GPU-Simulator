#pragma once

__global__ void WMat_rhs(myreal *rhs){
	let_idx_no_quit(idx,i,j,k);

	myreal ret = 0.;

	if(!is_pad(i,j,k))
		ret = ST.idt_2 *(4.*ST.W[idx]-ST.W_old[idx]) + fW(ST.phi_bar[idx],ST.W_bar[idx],ST.R_bar[idx]);
	rhs[idx] = ret;
}

__global__ void WMat_mul(myreal *x, myreal *y){
	let_idx_no_quit(idx,i,j,k);

	myreal ret =0.;

	bool pad = is_pad(i,j,k);
	if(!pad)
		ret = ST.idt3_2 * x[idx] - NP.LambdaW * laplacian(x,idx);
	y[idx] = ret;
	__syncthreads();
	if(pad)
		x[idx] = 0.;
}

void WMat::get_RHS(Vec &rhs) const {
	WMat_rhs<<<gp.dimGrid,gp.dimBlock>>>(cast(rhs));
	check("WMat::get_RHS");
}

void WMat::operator()(const Vec &x_, Vec &y) const {
	Vec &x = const_cast<Vec&>(x_);
	pad_reflex(x);

	WMat_mul<<<gp.dimGrid,gp.dimBlock>>>(cast(x),cast(y));
	check("WMat::mul");
}
