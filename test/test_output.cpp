//
// Created by wangj on 2024/9/4.
//
#include <complex>
#include <iostream>
#include <vector>
#include "lattice_desc.h"
#include "io/lqcd_read_write.h"
#include "qcu_parse_terminal.h"
#include <assert.h>
#include <mpi.h>
#include <stdexcept>
#include <H5Cpp.h>
using namespace std;

template <typename _Float>
void init_complex_vector(complex<_Float>* vec, int length) {
    for (int i = 0; i < length; ++i) {
        vec[i] = complex<_Float>(rand() % 20, rand() % 20);
    }
}

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
        const int Nc = Nc_required;

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

        // // 确保可以整除
        // if (Lt % nt != 0 || Lz % nz != 0 || Ly % ny != 0 || Lx % nx != 0) {
        //     if (rank == 0) {
        //         std::cerr << "错误：网格维度必须能被进程数整除\n";
        //     }
        //     MPI_Finalize();
        //     return 1;
        // }

        // 计算局部大小
        int local_lt = Lt_required / mpi_desc[T_DIM];
        int local_lz = Lz_required / mpi_desc[Z_DIM];
        int local_ly = Ly_required / mpi_desc[Y_DIM];
        int local_lx = Lx_required / mpi_desc[X_DIM];
        // const size_t local_lt = Lt / nt;
        // const size_t local_lz = Lz / nz;
        // const size_t local_ly = Ly / ny;
        // const size_t local_lx = Lx / nx;

        // 创建局部数组
        qcu::io::GaugeStorage<std::complex<double>> gauge(global_lattice_desc, Nc_required);

        // 初始化局部数据
        constexpr int MAX_ELEM = 9 * 32;
        int counter = 0;

        // 初始化数据
        std::complex<double>* data_ptr = gauge.data_ptr();
        for (size_t dim = 0; dim < 4; ++dim) {
            for (size_t t = 0; t < local_lt; ++t) {
                for (size_t z = 0; z < local_lz; ++z) {
                    for (size_t y = 0; y < local_ly; ++y) {
                        for (size_t x = 0; x < local_lx; ++x) {
                            for (size_t c1 = 0; c1 < Nc; ++c1) {
                                for (size_t c2 = 0; c2 < Nc; ++c2) {
                                    size_t one_dim_index = 
                                        dim * local_lt * local_lz * local_ly * local_lx 
                                        + t * local_lz * local_ly * local_lx 
                                        + z * local_ly * local_lx 
                                        + y * local_lx 
                                        + x;
                                    double temp = rank * 1000 + static_cast<double>(counter++);
                                    counter = counter % MAX_ELEM;
                                    data_ptr[one_dim_index * Nc * Nc + c1 * Nc + c2] = {temp, temp + 0.1};
                                }
                            }
                        }
                    }
                }
            }
        }

        const int Nd = 4;
        const int Lt = Lt_required;
        const int Lz = Lz_required;
        const int Ly = Ly_required;
        const int Lx = Lx_required;
        const int nx = mpi_desc[X_DIM];
        const int ny = mpi_desc[Y_DIM];
        const int nz = mpi_desc[Z_DIM];
        const int nt = mpi_desc[T_DIM];
        
        vector<int> dims = {Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2};
        printf("nx = %d, ny = %d, nz = %d, nt = %d\n", nx, ny, nz, nt);
        // const qcu::FourDimDesc mpi_desc(nx, ny, nz, nt);

        qcu::io::GaugeWriter<double> gauge_writer(rank, mpi_desc);
        gauge_writer.write("test_gauge.hdf5", gauge);

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