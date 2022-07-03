#include "state/state.h"

__constant__ NondimensionalizedParameters NP;
__constant__ StateBaseRaw ST;
StateBaseRaw st_raw;


void State::copy_NP_to_device() {
    cudaMemcpyToSymbol(::NP, &this->NP, sizeof(NP), 0, cudaMemcpyHostToDevice);
    check();
}
void State::copy_to_device() {
    st_raw.set(*this, dt);
    cudaMemcpyToSymbol(ST, &st_raw, sizeof(ST), 0, cudaMemcpyHostToDevice);
    check();
}



#include "state/reactive_terms.inl"
//#include "state/preconditioner.inl"
//#include "phiCN_mat.inl"
#include "phiBDF2_mat.inl"
#include "W_mat.inl"
#include "R_mat.inl"
#include "EQ.inl"
