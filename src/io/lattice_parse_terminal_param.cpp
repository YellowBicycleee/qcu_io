#include "qcu_parse_terminal.h"
#include <cstdio>
#include <getopt.h>
#include <vector>
#include <string>
#include <sstream>

static void print_usage(const char* arg) {
    fprintf(stderr, "Usage: %s --mpi mpi_str --lattice latt_size --Nc c --mIn m [-f|--file filename] \n", arg);
    fprintf(stderr, "          --mpi  x.y.z.t         how many mpi in each direction, example 1.1.1.1\n");
    fprintf(stderr, "          --lattice  x.y.z.t     how many lattice in each direction, example 4.4.4.4\n");
    fprintf(stderr, "          --Nc  c                how many color, example 3\n");
    fprintf(stderr, "          --mIn m                how many mIn, example 12\n");
    fprintf(stderr, " optional --file filename        file name\n");
}

static std::vector<int32_t> split_integers(const std::string &s, char delim) {
    std::vector<int32_t> elems;
    std::istringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(stoi(item));
    }
    return elems;
}

QcuHeader get_lattice_config(int argc, char *argv[], std::string& file, MPI_Desc& mpi_desc) {
    QcuHeader lattice_config;

    bool lattice_setted = false;
    bool mpi_setted = false;
    bool Nc_setted = false;
    bool mIn_setted = false;
    int Nc = 0;
    int mInput = 0;

    int opt;                // getopt_long() 的返回值
    int digit_optind = 0; // 设置短参数类型及是否需要参数

    int option_index = 0;
    const char *optstring = "m:l:c:i:";

    static struct option long_options[] = {
        {"mpi",     required_argument, NULL, 'm'},
        {"lattice", required_argument, NULL, 'l'},
        {"Nc",      required_argument, NULL, 'c'},
        {"mIn",     required_argument, NULL, 'i'},
        {"file",    required_argument, NULL, 'f'},
        {"help",    no_argument,       NULL, 'h'},
        {0, 0, 0, 0} 
    };

    while ( (opt = getopt_long(argc,
                               argv,
                               optstring,
                               long_options,
                               &option_index)) != -1) {
        switch (opt)
        {
        case 'm': // --mpi  or -m
            {
                std::vector<int32_t> mpiDesc = split_integers(std::string(optarg), '.');
                if (mpiDesc.size() != 4) {
                    fprintf(stderr, "mpiDesc dim must be 4, but now, dim is %lu\n", mpiDesc.size());
                    exit(EXIT_FAILURE);
                }

                mpi_desc.data[X_DIM] = mpiDesc[X_DIM];
                mpi_desc.data[Y_DIM] = mpiDesc[Y_DIM];
                mpi_desc.data[Z_DIM] = mpiDesc[Z_DIM];
                mpi_desc.data[T_DIM] = mpiDesc[T_DIM];
                mpi_setted = true;
            }
            break;
        case 'l': // --lattice or -l
            {
                std::vector<int32_t> lattice = split_integers(std::string(optarg), '.');
                if (lattice.size() != 4) {
                    fprintf(stderr, "Lattice dim must be 4, but now, dim is %lu\n", lattice.size());
                    exit(EXIT_FAILURE);
                }
                lattice_config.m_lattice_desc.data[X_DIM] = lattice[X_DIM];
                lattice_config.m_lattice_desc.data[Y_DIM] = lattice[Y_DIM];
                lattice_config.m_lattice_desc.data[Z_DIM] = lattice[Z_DIM];
                lattice_config.m_lattice_desc.data[T_DIM] = lattice[T_DIM];
                lattice_setted = true;
            }
            break;
        case 'c': // --Nc or -c
            {
                Nc = stoi(std::string(optarg));
                lattice_config.m_Nc = Nc;
                Nc_setted = true;
            }
            break;
        case 'i': // --mIn or -i
            {
                mInput = stoi(std::string(optarg));
                lattice_config.m_MInput = mInput;
                mIn_setted = true;
            }
            break;
        case 'f': // --file or -f
            {
                file = std::string(optarg);
            }
            break;
        case 'h': // --help or -h
            {
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            }
            break;
        default:
            {
                fprintf(stderr, "unknown opt = %c\n", opt); // 命令参数，亦即 -a -b -n -r
                fprintf(stderr, "optarg = %s\n", optarg); // 参数内容
                fprintf(stderr, "optind = %d\n", optind); // 下一个被处理的下标值
                fprintf(stderr, "argv[optind - 1] = %s\n",  argv[optind - 1]); // 参数内容
                fprintf(stderr, "option_index = %d\n", option_index);  // 当前打印参数的下标值
                fprintf(stderr, "\n");
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        }
    }

    if (!lattice_setted || !mpi_setted || !Nc_setted || !mIn_setted) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return lattice_config;
}