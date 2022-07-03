#pragma once
#include "grid.cuda/gallery.h"
#include <cusp/blas.h>
#include <curand.h>
#include <curand_kernel.h>



#define let_xyz(x,y,z,d)				\
	myreal x = NULL;	\
myreal y = (j-Npad+.5+(d==1?.5:0.)) * GP.dy;	\
myreal z = (k-Npad+.5+(d==2?.5:0.)) * GP.dz;

#define let_gallery(idx,i,j,k,x,y,z)		\
	let_idx(idx,i,j,k);				\
let_xyz(x,y,z,3)				\
if (!GP.is_3d)				\
x = myreal(0.5);

struct axpb_functor {
	const myreal a, b;
	axpb_functor(myreal a_, myreal b_) : a(a_), b(b_){}
	__host__ __device__
		myreal operator()(const myreal& x) const {return a * x + b;}
};



__device__ inline myreal threshold_compact(myreal ret, myreal width) {
	ret /= width;
	if (ret > PI/2)
		return 1.;
	else if (ret < -PI/2)
		return 0.;
	ret = tanh(tan(ret));
	return (1+ret)/2.;
}

__device__ inline myreal threshold(myreal ret, myreal width) {
	return threshold_compact(ret, width);
}



namespace gallery {

	__global__
		void scal(myreal *v, myreal a) {
			let_idx(idx,i,j,k);
			v[idx] *= a;
		}

	__global__
		void constant(myreal *v, myreal a) {
			let_idx(idx,i,j,k);
			v[idx] = a;
		}

	__global__ void set_randstate(curandState *state0, const int iid){
		let_gallery(idx,i,j,k,x,y,z);
		//curand_init(2337,idx,0,&state0[idx]);
		curand_init(iid,idx,0,&state0[idx]);
	}

	__global__ void add_yang(myreal *v, curandState *state, myreal phi0=0.55, myreal osc = 0.001){
		let_gallery(idx,i,j,k,x,y,z);
		myreal ret = 0.;
		ret = curand_uniform(&state[idx]);
		v[idx] = osc*ret + phi0;
	}

	__global__ void add_yang_transform(myreal *v, curandState *state, myreal phi0=0.55, myreal osc = 0.001){
		let_gallery(idx,i,j,k,x,y,z);
		myreal ret = 0.;
		ret = curand_uniform(&state[idx]);
		v[idx] = osc*ret*(ret > 0.25) + phi0;
	}

	__global__ void add_square(myreal *v, curandState *state, myreal phi0, myreal osc, myreal R,  myreal width){
		let_gallery(idx,i,j,k,x,y,z);
		myreal ret = 0.;
		myreal dist = max(fabs(y-0.5*GP.Ly),fabs(z-0.5*GP.Lz)); //sqrt(pow2(y-0.5*GP.Ly)+pow2(z-0.5*GP.Lz));
		ret = threshold(R-dist,width) * ( osc*curand_uniform(&state[idx])+phi0);
		//v[idx] = 2*ret-1.;
		v[idx] = ret;
	}

	__global__ void add_crossing(myreal *v, myreal R1, myreal R2, myreal width){
		let_gallery(idx,i,j,k,x,y,z);
		myreal ret = 0.;

		/*
		   myreal y0 = 0.5 * GP.Ly;
		   myreal z0 = 0.5 * GP.Lz;
		   myreal dist1 = fabs(y-y0);
		   myreal dist2 = fabs(z-z0);
		   v[idx] = min( threshold(R1-dist1,width), threshold(R2-dist2,width));
		   v[idx] = max(v[idx], min( threshold(R1-dist2,width), threshold(R2-dist1,width)));
		   */
		ret = threshold(y-0.25*GP.Ly,width) * threshold(0.75*GP.Ly-y,width)* (max(threshold(z-0.75*GP.Lz,width),threshold(0.25*GP.Lz-z,width)));
		//ret = min(ret,threshold(0.75*GP.Ly-y,width) * threshold(0.75*GP.Lz-z,width));
		v[idx] = 2.*ret-1.;
	}


	void set(myreal *v, const std::string &shape, myreal phi_max, myreal width) {
		std::string name;
		std::istringstream ss(shape);
		ss>>name;

		if(name == "Random"){
			std::cout<<"Init Yang Profile \n";
			myreal phi0 = next_arg(ss,0.55);
			myreal osc  = next_arg(ss,0.001);

			curandState *devState;
			cudaMalloc(&devState,gp.Ndata*sizeof(curandState));
			set_randstate<<<gp.dimGrid,gp.dimBlock>>>(devState,2137);
			add_yang<<<gp.dimGrid,gp.dimBlock>>>(v,devState,phi0,osc);
			//add_yang_transform<<<gp.dimGrid,gp.dimBlock>>>(v,devState,phi0,osc);
		}else if(name == "Random2"){
			std::cout<<"Init Yang Profile \n";
			myreal phi0 = next_arg(ss,0.55);
			myreal osc  = next_arg(ss,0.001);

			curandState *devState;
			cudaMalloc(&devState,gp.Ndata*sizeof(curandState));
			set_randstate<<<gp.dimGrid,gp.dimBlock>>>(devState,2137);
			add_yang_transform<<<gp.dimGrid,gp.dimBlock>>>(v,devState,phi0,osc);
		}else if (name=="zero") {
			constant <<<gp.dimGrid,gp.dimBlock>>> (v, myreal(0));
			return;
		}
		else if(name == "Square"){
			myreal R = next_arg(ss,0.1);
			myreal phi0 = 0.;
			myreal osc  = 1.;


			curandState *devState;
			cudaMalloc(&devState,gp.Ndata*sizeof(curandState));
			set_randstate<<<gp.dimGrid,gp.dimBlock>>>(devState,2137);
			//add_yang<<<gp.dimGrid,gp.dimBlock>>>(v,devState,phi0,osc);
			add_square<<<gp.dimGrid,gp.dimBlock>>>(v,devState,phi0,osc,R,width);
		}
		else
			my_exit(-1,"shape unknown \n");
		/*{
		  std::cout<<"Crossing profile \n";
		  myreal R1 = next_arg(ss,0.1*gp.Ly);
		  myreal R2 = next_arg(ss,0.4*gp.Ly);
		  add_crossing<<<gp.dimGrid,gp.dimBlock>>>(cast(v),R1,R2,0.02*gp.Lz);
		//add_crossing<<<gp.dimGrid,gp.dimBlock>>>(cast(v),R1,R2,width);
		}
		*/
	}
	void set(Vec &v, const std::string &shape, myreal phi_max, myreal width, int i) {
		gallery::set(cast(v) + i*gp.Ndata, shape, phi_max, width);
	}

}
