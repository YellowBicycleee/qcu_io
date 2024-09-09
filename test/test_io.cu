//
// Created by wangj on 2024/9/4.
//
#include <iostream>
#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <complex>
#include <vector>
// #include <cuda_runtime_api.h>
#include "check_cuda.h"

using namespace std;

// __global__ void print (complex<double>* vec, int length) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("PRINT DEBUG: \n");
//         for (int i = 0; i < length; ++i) {
//             printf("i = %d, real = %e, imag = %e\n", i, vec[i], imag(vec[i]));
//         }
//         printf("\n");
//     }
// }

template <typename _Float>
void init_complex_vector(complex<_Float>* vec, int length) {
    static int init = 1;
    for (int i = 0; i < length; ++i) {
        vec[i] = complex<_Float>(init, init);
    }
    init++;
}

int main(int argc, char *argv[]) {

    int Lx = 2;
    int Ly = 2;
    int Lz = 1;
    int Lt = 1;
    int Ns = 1;
    int Nc = 1;
    int Nd = 4;
    int mInput = 1;

    LatticeConfig lattice_config {
        .m_Lx = Lx,
        .m_Ly = Ly,
        .m_Lz = Lz,
        .m_Lt = Lt,
        .m_Ns = Ns,
        .m_Nd = Nd,
        .m_Nc = Nc,
        .m_MInput = mInput,
        .m_data_format    = DataFormat::QUDA_FORMAT,
        .m_mrhs_shuffled  = MrhsShuffled::MRHS_SHUFFLED_NO,
        .m_src_rw_flag   = ReadWriteFlag::RW_YES,
        .m_dst_rw_flag   = ReadWriteFlag::RW_YES,
        .m_gauge_rw_flag = ReadWriteFlag::RW_YES
    };
    const char* file_name = "test_io.bin";
    auto mrhs_colorspinor_len = lattice_config.MrhsColorSpinorLength();
    auto gauge_len = lattice_config.GaugeLenth();

    complex<double>* h_src_ptr;
    complex<double>* h_dst_ptr;
    complex<double>* h_gauge_ptr;

    h_src_ptr   = new complex<double>[mrhs_colorspinor_len];
    h_dst_ptr   = new complex<double>[mrhs_colorspinor_len];
    h_gauge_ptr = new complex<double>[gauge_len];

    cout << "mrhs_colorspinor_len = " << mrhs_colorspinor_len << endl;
    cout << "gauge_len = "            << gauge_len << endl;

    init_complex_vector(h_src_ptr, mrhs_colorspinor_len);
    init_complex_vector(h_dst_ptr, mrhs_colorspinor_len);
    init_complex_vector(h_gauge_ptr, gauge_len);

    // print
    cout << "src: " << endl;
    for (int i = 0; i < mrhs_colorspinor_len; ++i) {
        cout << h_src_ptr[i] << endl;
    }
    cout << "dst: " << endl;
    for (int i = 0; i < mrhs_colorspinor_len; ++i) {
        cout << h_dst_ptr[i] << endl;
    }
    cout << "gauge: " << endl;
    for (int i = 0; i < gauge_len; ++i) {
        cout << h_gauge_ptr[i] << endl;
    }



    // DEBUG 
    cout << h_dst_ptr[0] << endl;
    cout << h_src_ptr[0] << endl;
    // END DEBUG
    complex<double>* d_src_ptr;
    complex<double>* d_dst_ptr;
    complex<double>* d_gauge_ptr;

    CHECK_CUDA(cudaMalloc (&d_src_ptr,   sizeof(complex<double>) * mrhs_colorspinor_len));
    CHECK_CUDA(cudaMalloc (&d_dst_ptr,   sizeof(complex<double>) * mrhs_colorspinor_len));
    CHECK_CUDA(cudaMalloc (&d_gauge_ptr, sizeof(complex<double>) * gauge_len));

    CHECK_CUDA(cudaMemcpy(d_src_ptr,   h_src_ptr,   sizeof(complex<double>) * mrhs_colorspinor_len, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dst_ptr,   h_dst_ptr,   sizeof(complex<double>) * mrhs_colorspinor_len, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gauge_ptr, h_gauge_ptr, sizeof(complex<double>) * gauge_len,            cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    // template <typename _FloatType>
    write_to_file<double> (d_src_ptr, d_dst_ptr, d_gauge_ptr, file_name, lattice_config);

    CHECK_CUDA(cudaFree (d_src_ptr));
    CHECK_CUDA(cudaFree (d_dst_ptr));
    CHECK_CUDA(cudaFree (d_gauge_ptr));
    
    delete[] h_src_ptr;
    delete[] h_dst_ptr;
    delete[] h_gauge_ptr;
}