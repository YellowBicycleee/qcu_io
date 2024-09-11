//
// Created by wangj on 2024/9/4.
//
#include <iostream>
#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <complex>
#include <vector>
#include "check_cuda.h"

using namespace std;


template <typename _Float>
void init_complex_vector(complex<_Float>* vec, int length) {
    static int init = 1;
    for (int i = 0; i < length; ++i) {
        vec[i] = complex<_Float>(init, init);
    }
    init++;
}


int main(int argc, char *argv[]) {


    int Ns = 4;
    int Nd = 4;

    string file_path = "test_io.bin";
    LatticeConfig lattice_config = get_lattice_config(argc, argv, file_path);
    lattice_config.m_Ns = Ns;
    lattice_config.m_Nd = Nd;
    
    lattice_config.m_data_format   = DataFormat::QUDA_FORMAT;
    lattice_config.m_mrhs_shuffled = MrhsShuffled::MRHS_SHUFFLED_NO;
    lattice_config.m_io_position   = IOPosition::IO_FERMION_IN | IOPosition::IO_FERMION_OUT | IOPosition::IO_GAUGE;


    // const char* file_name = "test_io.bin";
    auto mrhs_colorspinor_len = lattice_config.MrhsColorSpinorLength();
    auto gauge_len = lattice_config.GaugeLength();

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
    // cout << "src: " << endl;
    // for (int i = 0; i < mrhs_colorspinor_len; ++i) {
    //     cout << h_src_ptr[i] << endl;
    // }
    // cout << "dst: " << endl;
    // for (int i = 0; i < mrhs_colorspinor_len; ++i) {
    //     cout << h_dst_ptr[i] << endl;
    // }
    // cout << "gauge: " << endl;
    // for (int i = 0; i < gauge_len; ++i) {
    //     cout << h_gauge_ptr[i] << endl;
    // }

    // DEBUG 
    // cout << h_dst_ptr[0] << endl;
    // cout << h_src_ptr[0] << endl;
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

    write_to_file<double> (d_src_ptr, d_dst_ptr, d_gauge_ptr, file_path, lattice_config);

    cout << "mrhs_colorspinor_len = " << mrhs_colorspinor_len << endl;
    cout << "gauge_len = "            << gauge_len << endl;
    cout << "sizeof config = "        << sizeof(LatticeConfig) << endl;
    CHECK_CUDA(cudaFree (d_src_ptr));
    CHECK_CUDA(cudaFree (d_dst_ptr));
    CHECK_CUDA(cudaFree (d_gauge_ptr));
    

    FILE* debugfile = fopen("file_output_data.txt", "w");
    if (!debugfile) {
        fprintf(stderr, "failed to open file file_output_data.txt\n");
    }
    fprintf(debugfile, "================== GAUGE: ======================\n");
    for (int i = 0; i < gauge_len; ++i) {
        fprintf(debugfile, "(%lf, %lf) ", h_gauge_ptr[i].real(), h_gauge_ptr[i].imag());
        if (i % 5 == 0) {
            fprintf(debugfile, "\n");
        }
    }
    fprintf(debugfile, "\n==================fermion_in ===================\n");
    for (int i = 0; i < mrhs_colorspinor_len; ++i) {
        fprintf(debugfile, "(%lf, %lf) ", h_src_ptr[i].real(), h_src_ptr[i].imag());
        if (i % 5 == 0) {
            fprintf(debugfile, "\n");
        }
    }
    fprintf(debugfile, "\n==================fermion_out ===================\n");
    for (int i = 0; i < mrhs_colorspinor_len; ++i) {
        fprintf(debugfile, "(%lf, %lf) ", h_dst_ptr[i].real(), h_dst_ptr[i].imag());
        if (i % 5 == 0) {
            fprintf(debugfile, "\n");
        }
    }
    fclose(debugfile);

    delete[] h_src_ptr;
    delete[] h_dst_ptr;
    delete[] h_gauge_ptr;
}