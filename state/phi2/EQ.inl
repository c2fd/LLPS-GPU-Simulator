#pragma once
__global__ void partial_q_kernel(myreal *rhs, myreal *phi){
	let_idx_no_quit(idx,i,j,k);

	myreal ret = 0.;
	if(!is_pad(i,j,k)){
		ret = partial_q_device(phi[idx]);
	}
	rhs[idx] = ret;
}

__global__ void init_qq_kernel(myreal *myqq, const myreal *myphi){
	let_idx_no_quit(idx,i,j,k);

	myreal ret = 0.;                
	if(!is_pad(i,j,k)){                     
		ret = init_q_device(myphi[idx]);
	}
	myqq[idx] = ret;
}

__global__ void update_qq_BDF2_kernel(myreal *qq_np1, const myreal *phi_np1, const myreal *phi, const myreal *phi_old, const myreal *qq, const myreal *qq_old, const myreal *parqq){
	let_idx_no_quit(idx,i,j,k);
	myreal ret = 0.;

	if(!is_pad(i,j,k))
		ret +=  parqq[idx] * (phi_np1[idx] - 4./3. * phi[idx] + 1./3.*phi_old[idx]) + 4./3.*qq[idx] - 1./3.*qq_old[idx];
	qq_np1[idx] = ret;
}

void State::update_qq(){
	Vec qq_bar(gp.Ndata,0.);
	update_qq_BDF2_kernel<<<gp.dimGrid,gp.dimBlock>>>(cast(qq_bar),cast(phi_np1),cast(phi),cast(phi_old),cast(qq),cast(qq_old),cast(parqq));
	pad_phi(qq_bar);

	blas::copy(qq,qq_old);
	blas::copy(qq_bar,qq);
}

void State::init_qq(Vec &myqq, const Vec &myphi){
	init_qq_kernel<<<gp.dimGrid,gp.dimBlock>>>(cast(myqq),cast(myphi));
	pad_phi(myqq);
	std::cout<<"max,min,sum \t "<<max(myqq)<<" \t "<<min(myqq)<<" \t "<<sum(myqq)<<" \n"; 
}

void State::update_parqq(){
	partial_q_kernel<<<gp.dimGrid,gp.dimBlock>>>(cast(parqq),cast(phi_bar));
	pad_phi(parqq);
}
