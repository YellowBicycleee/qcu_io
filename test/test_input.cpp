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
#include <numeric>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        // constexpr int Lx_required = 4;
        // constexpr int Ly_required = 4;
        // constexpr int Lz_required = 4;
        // constexpr int Lt_required = 4;
        constexpr int Lx_required = 2;
        constexpr int Ly_required = 2;
        constexpr int Lz_required = 2;
        constexpr int Lt_required = 2;
        constexpr int Nc_required = 3;

        std::vector<int> global_lattice_desc = {Lx_required, Ly_required, Lz_required, Lt_required};
        
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
        std::istringstream iss(grid_str);
        std::vector<int> mpi_desc;
        
        while (getline(iss, grid_str, '.')) {
            mpi_desc.push_back(std::stoi(grid_str));
        }
        assert(mpi_desc.size() == global_lattice_desc.size());

        if (std::accumulate(mpi_desc.begin(), mpi_desc.end(), 1, std::multiplies<int>()) != size) {
            if (rank == 0) {
                std::cerr << "错误：进程网格大小 (" << std::endl;
                for (int i = 0; i < mpi_desc.size() - 1; ++i) {
                    std::cerr << mpi_desc[i] << ",";
                }
                std::cerr << mpi_desc[mpi_desc.size() - 1] << ") 与总进程数 " << size << " 不匹配\n";
            }
            MPI_Finalize();
            return 1;
        }

        qcu::io::GaugeStorage<std::complex<double>> gauge(global_lattice_desc, Nc_required);

        qcu::io::GaugeReader<double> reader(rank, mpi_desc);
        reader.read("test_gauge.hdf5", gauge);

        int Nc = gauge.get_n_color();
        int Lt = gauge.get_global_lattice_desc()[T_DIM];
        int Lz = gauge.get_global_lattice_desc()[Z_DIM];
        int Ly = gauge.get_global_lattice_desc()[Y_DIM];
        int Lx = gauge.get_global_lattice_desc()[X_DIM];

        int local_lt = Lt / mpi_desc[T_DIM];
        int local_lz = Lz / mpi_desc[Z_DIM];
        int local_ly = Ly / mpi_desc[Y_DIM];
        int local_lx = Lx / mpi_desc[X_DIM];

        if (rank == 0) {
            std::cout << "Nc = " << Nc << ", Lt = " << Lt << ", Lz = " << Lz << ", Ly = " << Ly << ", Lx = " << Lx << std::endl;
            std::cout << "local_lt = " << local_lt << ", local_lz = " << local_lz << ", local_ly = " << local_ly << ", local_lx = " << local_lx << std::endl;
        }


        // 按进程顺序输出每个进程的两个Nc * Nc矩阵
        for (int current_rank = 0; current_rank < size; ++current_rank) {
            if (rank == current_rank && global_lattice_desc.size() == 4) {
                std::cout << "\n进程 " << rank << " 的两个 Nc * Nc 矩阵:\n";
                // 固定其他维度,输出两个color矩阵
                const int t = 0, z = 0, y = 0, x = 0;
                const int vol = local_lx * local_ly * local_lz * local_lt;
                int one_dim_index = ((t * local_lz + z) * local_ly + y) * local_lx + x;
                std::cout << "one_dim_index = " << one_dim_index << std::endl;
                std::complex<double> *data_ptr = gauge.data_ptr();
                // 输出前两个维度
                for (size_t dim = 0; dim < 2; ++dim) {
                    std::cout << "维度 " << dim << ":\n";
                    for (size_t c1 = 0; c1 < Nc; ++c1) {
                        for (size_t c2 = 0; c2 < Nc; ++c2) {
                            std::cout << data_ptr[(dim * vol + one_dim_index) * Nc * Nc + c1 * Nc + c2] << "\t";
                        }
                        std::cout << "\n";
                    }
                    std::cout << "\n";
                }
                std::cout << std::flush;

                {
                    int t = local_lt - 1;
                    int z = local_lz - 1;
                    int y = local_ly - 1;
                    int x = local_lx - 1;
                    int one_dim_index = ((t * local_lz + z) * local_ly + y) * local_lx + x;

                    std::cout << "last matrix " << ":\n";
                    for (size_t c1 = 0; c1 < Nc; ++c1) {
                        for (size_t c2 = 0; c2 < Nc; ++c2) {
                            std::cout << data_ptr[(3 * vol + one_dim_index) * Nc * Nc + c1 * Nc + c2] << "\t";
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