#pragma once
#include "grid.cuda/grid_params.h"
#include "grid.cuda/debug.h"


__constant__ GridParamsBase GP;


void GridParams::copy_to_device() const {
    cudaMemcpyToSymbol(GP, this, sizeof(GridParamsBase), 0,
		       cudaMemcpyHostToDevice);
    check();
}


inline __device__
bool is_periodic(int d) {
    return GP.periodicity & (1<<d);
}
