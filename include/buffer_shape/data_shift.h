#pragma once
#include "base/datatype/qcu_complex.cuh"
namespace qcu {
// m, n 是原始形状
// sunwEO[Nc * Nc, 4 * vol] -> qudaEO[4 * vol, Nc * Nc]
template <typename _Float>
void gauge_from_sunwEO_to_qudaEO(Complex<_Float>* qudaEO_guage, Complex<_Float>* sunwEO_gauge, int m, int n);

template <typename _Float>
void from_qudaEO_to_sunwEO(Complex<_Float>* sunwEO_gauge, Complex<_Float>* qudaEO_gauge, int m, int n);

}