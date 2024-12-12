//
// Created by wangj on 2024/9/4.
//
#include <complex>
#include <iostream>
#include <vector>
#include "lattice_desc.h"
#include "io/lqcd_read_write.h"
#include "qcu_parse_terminal.h"
#include <cassert>
#include <mpi.h>
#include <H5Cpp.h>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        constexpr int Lx_required = 4;
        constexpr int Ly_required = 4;
        constexpr int Lz_required = 4;
        constexpr int Lt_required = 4;
        constexpr int Nc_required = 3;
        vector<int> dims = {4, Lt_required, Lz_required, Ly_required, Lx_required, Nc_required, Nc_required * 2};

        // 解析命令行参数 (x,y,z,t顺序)
        if (argc != 2) {
            if (rank == 0) {
                std::cerr << "用法: mpirun -n <进程数> " << argv[0] << " Nx.Ny.Nz.Nt\n";
            }
            MPI_Finalize();
            return 1;
        }

        // 解析进程网格
        std::string grid_str(argv[1]);
        int nx, ny, nz, nt;
        size_t pos = 0;
        nx = std::stoi(grid_str, &pos); grid_str = grid_str.substr(pos + 1);
        ny = std::stoi(grid_str, &pos); grid_str = grid_str.substr(pos + 1);
        nz = std::stoi(grid_str, &pos); grid_str = grid_str.substr(pos + 1);
        nt = std::stoi(grid_str);

        if (nx * ny * nz * nt != size) {
            if (rank == 0) {
                std::cerr << "错误：进程网格大小 (" << nx << "," << ny << "," 
                         << nz << "," << nt << ") 与总进程数 " << size << " 不匹配\n";
            }
            MPI_Finalize();
            return 1;
        }

        qcu::io::Gauge4Dim<std::complex<double>> gauge(0, 0, 0, 0, 0);

        qcu::FourDimDesc mpi_desc(nx, ny, nz, nt);
        qcu::io::GaugeReader<double> reader(0, mpi_desc);
        reader.read("test_gauge.hdf5", dims, gauge);

        int Nc = gauge.get_Nc();
        int Lt = gauge.get_Lt();
        int Lz = gauge.get_Lz();
        int Ly = gauge.get_Ly();
        int Lx = gauge.get_Lx();

        int local_lt = Lt / nt;
        int local_lz = Lz / nz;
        int local_ly = Ly / ny;
        int local_lx = Lx / nx;

        if (rank == 0) {
            std::cout << "Nc = " << Nc << ", Lt = " << Lt << ", Lz = " << Lz << ", Ly = " << Ly << ", Lx = " << Lx << std::endl;
            std::cout << "local_lt = " << local_lt << ", local_lz = " << local_lz << ", local_ly = " << local_ly << ", local_lx = " << local_lx << std::endl;
        }


        // 按进程顺序输出每个进程的两个Nc * Nc矩阵
        for (int current_rank = 0; current_rank < size; ++current_rank) {
            if (rank == current_rank) {
                std::cout << "\n进程 " << rank << " 的两个 Nc * Nc 矩阵:\n";
                // 固定其他维度,输出两个color矩阵
                const size_t t = 0, z = 0, y = 0, x = 0;
                // 输出前两个维度
                for (size_t dim = 0; dim < 2; ++dim) {
                    std::cout << "维度 " << dim << ":\n";
                    for (size_t c1 = 0; c1 < Nc; ++c1) {
                        for (size_t c2 = 0; c2 < Nc; ++c2) {
                            std::cout << gauge(dim, t, z, y, x, c1, c2) << "\t";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
                std::cout << std::flush;

                {
                    std::cout << "last matrix " << ":\n";
                    for (size_t c1 = 0; c1 < Nc; ++c1) {
                        for (size_t c2 = 0; c2 < Nc; ++c2) {
                            std::cout << gauge(3, local_lt - 1, local_lz - 1, local_ly - 1, local_lx - 1, c1, c2) << "\t";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
            }
            MPI_Barrier(MPI_COMM_WORLD); // 确保按顺序输出
        }

        MPI_Finalize();
        return 0;

    } catch (const H5::Exception& e) {
        if (rank == 0) {
            std::cerr << "HDF5错误：" << e.getCDetailMsg() << '\n';
        }
        MPI_Finalize();
        return 1;
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "错误：" << e.what() << '\n';
        }
        MPI_Finalize();
        return 1;
    }
}