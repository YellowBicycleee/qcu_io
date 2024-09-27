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
#include "qcu_parse_terminal.h"
#include <cassert>
using namespace std;

int main(int argc, char* argv[]) {
  // int Ns = 4;
  // int Nd = 4;
  TerminalConfig config;
  get_lattice_config(argc, argv, config);
  config.detail();

  assert(config.fermion_in_file_configured == 1 && config.gauge_in_file_configured == 1);
  cout << "config.fermion_in_file = " << config.fermion_in_file << endl;
  cout << "config.gauge_in_file = " << config.gauge_in_file << endl;
  QcuHeader fermionConfig;
  fermionConfig.m_data_format       = DataFormat::QUDA_FORMAT;
  fermionConfig.m_storage_precision = StoragePrecision::PRECISION_DOUBLE;
  fermionConfig.m_mrhs_shuffled     = MrhsShuffled::MRHS_SHUFFLED_NO;
  fermionConfig.m_storage_type      = StorageType::TYPE_FERMION;
  fermionConfig.m_lattice_desc      = config.lattice_desc;
  fermionConfig.m_Nc     = config.Nc;
  fermionConfig.m_MInput = config.mInput;
  fermionConfig.m_Ns     = Ns;

  QcuHeader gaugeConfig;
  gaugeConfig.m_data_format       = DataFormat::QUDA_FORMAT;
  gaugeConfig.m_storage_precision = StoragePrecision::PRECISION_DOUBLE;
  gaugeConfig.m_mrhs_shuffled     = MrhsShuffled::MRHS_SHUFFLED_NO;
  gaugeConfig.m_storage_type      = StorageType::TYPE_GAUGE;
  gaugeConfig.m_lattice_desc      = config.lattice_desc;
  gaugeConfig.m_Nc     = config.Nc;
  gaugeConfig.m_Ngauge = 1;
  gaugeConfig.m_Nd     = Nd;


  auto mrhs_colorspinor_len = fermionConfig.MrhsColorSpinorLength();
  auto gauge_len = gaugeConfig.GaugeLength();

  complex<double>* h_dst_ptr;
  complex<double>* h_gauge_ptr;

  h_dst_ptr   = new complex<double>[mrhs_colorspinor_len];
  h_gauge_ptr = new complex<double>[gauge_len];

  cout << "mrhs_colorspinor_len = " << mrhs_colorspinor_len << endl;
  cout << "gauge_len = " << gauge_len << endl;

  MPI_Coordinate coord;
  GaugeReader<double> gaugeReader(config.gauge_in_file, 
                                  gaugeConfig, 
                                  config.mpi_desc, 
                                  coord);

  gaugeReader.read_gauge(h_gauge_ptr, 0);
  // gaugeWriter.write_gauge(h_gauge_ptr);

  FermionReader<double> fermionReader(config.fermion_in_file, 
                                      fermionConfig, 
                                      config.mpi_desc, 
                                      coord);
  fermionReader.read_fermion(h_dst_ptr, 0);


  FILE* debugfile = fopen("file_input_data.txt", "w");
  if (!debugfile) {
    fprintf(stderr, "failed to open file file_output_data.txt\n");
  }
  fprintf(debugfile, "================== GAUGE: ======================\n");
  for (int i = 0; i < gauge_len; ++i) {
    fprintf(debugfile, "(%lf, %lf) ", h_gauge_ptr[i].real(), h_gauge_ptr[i].imag());
    if (i != 0 && i % 5 == 0) {
      fprintf(debugfile, "\n");
    }
  }
  // fprintf(debugfile, "\n==================fermion_in ===================\n");
  // for (int i = 0; i < mrhs_colorspinor_len; ++i) {
  //   fprintf(debugfile, "(%lf, %lf) ", h_src_ptr[i].real(), h_src_ptr[i].imag());
  //   if (i != 0 && i % 5 == 0) {
  //     fprintf(debugfile, "\n");
  //   }
  // }
  fprintf(debugfile, "\n==================fermion ===================\n");
  for (int i = 0; i < mrhs_colorspinor_len; ++i) {
    fprintf(debugfile, "(%lf, %lf) ", h_dst_ptr[i].real(), h_dst_ptr[i].imag());
    if (i != 0 && i % 5 == 0) {
      fprintf(debugfile, "\n");
    }
  }
  fclose(debugfile);
  // cout << "mrhs_colorspinor_len = " << mrhs_colorspinor_len << endl;
  // cout << "gauge_len = " << gauge_len << endl;
  // std::cout << "sizeof(Config) = " << sizeof(QcuHeader) << std::endl;


  // delete[] h_src_ptr;
  delete[] h_dst_ptr;
  delete[] h_gauge_ptr;
}