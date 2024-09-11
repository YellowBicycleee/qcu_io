//
// Created by wjc on 2024/9/11.
//
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "check_cuda.h"
#include "lattice_desc.h"
#include "lqcd_read_write.h"

using namespace std;

int main(int argc, char* argv[]) {
  int Ns = 4;
  int Nd = 4;

  string file_path = "test_io.bin";
  LatticeConfig lattice_config = get_lattice_config(argc, argv, file_path);
  lattice_config.m_Ns = Ns;
  lattice_config.m_Nd = Nd;

  lattice_config.m_data_format = DataFormat::QUDA_FORMAT;
  lattice_config.m_mrhs_shuffled = MrhsShuffled::MRHS_SHUFFLED_NO;
  lattice_config.m_io_position =
      IOPosition::IO_FERMION_IN | IOPosition::IO_FERMION_OUT | IOPosition::IO_GAUGE;

  // const char* file_name = "test_io.bin";
  auto mrhs_colorspinor_len = lattice_config.MrhsColorSpinorLength();
  auto gauge_len = lattice_config.GaugeLength();

  complex<double>* h_src_ptr;
  complex<double>* h_dst_ptr;
  complex<double>* h_gauge_ptr;

  h_src_ptr = new complex<double>[mrhs_colorspinor_len];
  h_dst_ptr = new complex<double>[mrhs_colorspinor_len];
  h_gauge_ptr = new complex<double>[gauge_len];

  cout << "mrhs_colorspinor_len = " << mrhs_colorspinor_len << endl;
  cout << "gauge_len = " << gauge_len << endl;

  complex<double>* d_src_ptr;
  complex<double>* d_dst_ptr;
  complex<double>* d_gauge_ptr;

  CHECK_CUDA(cudaMalloc(&d_src_ptr, sizeof(complex<double>) * mrhs_colorspinor_len));
  CHECK_CUDA(cudaMalloc(&d_dst_ptr, sizeof(complex<double>) * mrhs_colorspinor_len));
  CHECK_CUDA(cudaMalloc(&d_gauge_ptr, sizeof(complex<double>) * gauge_len));

  read_from_file<double>(d_src_ptr,    // 数据指针
                         d_dst_ptr,    // 数据指针
                         d_gauge_ptr,  // 数据指针
                         file_path,    // 文件路径
                         lattice_config);
  CHECK_CUDA(cudaMemcpy(h_src_ptr, d_src_ptr, sizeof(complex<double>) * mrhs_colorspinor_len,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_dst_ptr, d_dst_ptr, sizeof(complex<double>) * mrhs_colorspinor_len,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_gauge_ptr, d_gauge_ptr, sizeof(complex<double>) * gauge_len,
                        cudaMemcpyDeviceToHost));

  FILE* debugfile = fopen("file_input_data.txt", "w");
  fprintf(debugfile, "================== GAUGE: ======================\n");
  for (int i = 0; i < gauge_len; ++i) {
    fprintf(debugfile, "(%lf, %lf) ", h_gauge_ptr[i].real(), h_gauge_ptr[i].imag());
    if (i != 0 && i % 5 == 0) {
      fprintf(debugfile, "\n");
    }
  }
  fprintf(debugfile, "\n==================fermion_in ===================\n");
  for (int i = 0; i < mrhs_colorspinor_len; ++i) {
    fprintf(debugfile, "(%lf, %lf) ", h_src_ptr[i].real(), h_src_ptr[i].imag());
    if (i != 0 && i % 5 == 0) {
      fprintf(debugfile, "\n");
    }
  }
  fprintf(debugfile, "\n==================fermion_out ===================\n");
  for (int i = 0; i < mrhs_colorspinor_len; ++i) {
    fprintf(debugfile, "(%lf, %lf) ", h_dst_ptr[i].real(), h_dst_ptr[i].imag());
    if (i != 0 && i % 5 == 0) {
      fprintf(debugfile, "\n");
    }
  }
  fclose(debugfile);
  CHECK_CUDA(cudaFree(d_src_ptr));
  CHECK_CUDA(cudaFree(d_dst_ptr));
  CHECK_CUDA(cudaFree(d_gauge_ptr));

  delete[] h_src_ptr;
  delete[] h_dst_ptr;
  delete[] h_gauge_ptr;
}