/**
 * 2013-11-15 Preconditioner
 * p \nabla^4 u + q \nabla^2 u + u = g
 * coef \nabla^2 u + \lambda u = g
 * */

__global__ void SimpleTry(myreal *x, myreal *y)
{
	let_idx_no_quit(idx,i,j,k);
	bool pad = is_pad(i,j,k);
	myreal ret = myreal(0.);
	if(!pad)
	{
		ret = x[idx];
	}
	y[idx] = ret;
	__syncthreads();

	if(pad)
		x[idx] = myreal(0.);
}

void PrePhi::get_RHS(Vec &rhs) const{
}

void PrePhi::operator()(const Vec &x_, Vec &y) const{
	using namespace my_helmholtz;
	Vec &x = const_cast<Vec&>(x_);

	pad_reflex(x);
	SimpleTry<<<gp.dimGrid,gp.dimBlock>>>(cast(x),cast(y));
	my_helmholtz::init(PERIODIC,PERIODIC);
	my_helmholtz::solve_biharmonic(cast(y),sub_p,sub_q,sub_lam);
	check("PrePhi");
}
