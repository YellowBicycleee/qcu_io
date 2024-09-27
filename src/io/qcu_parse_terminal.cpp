#include "qcu_parse_terminal.h"
#include <cstdio>
#include <getopt.h>
#include <vector>
#include <string>
#include <sstream>

void TerminalConfig::detail() {
    fprintf(stdout, "================Terminal config: ===================\n");
    fprintf(stdout, "Nc = %d\n", Nc);
    fprintf(stdout, "mInput = %d\n", mInput);
    // MPI_DESC
    if (mpi_configured) {
        fprintf(stdout, "mpi_desc = %d.%d.%d.%d\n", mpi_desc.data[X_DIM], mpi_desc.data[Y_DIM], mpi_desc.data[Z_DIM], mpi_desc.data[T_DIM]);
    }
    else {
        fprintf(stdout, "mpi_desc is not configured\n");
    }
    // LATTICE_DESC
    if (lattice_configured) {
        fprintf(stdout, "lattice_desc = %d.%d.%d.%d\n", lattice_desc.data[X_DIM], lattice_desc.data[Y_DIM], lattice_desc.data[Z_DIM], lattice_desc.data[T_DIM]);
    }
    else {
        fprintf(stdout, "lattice_desc is not configured\n");
    }
    // GAUGE_IN_FILE name
    if (gauge_in_file_configured) {
        fprintf(stdout, "gauge_in_file = %s\n", gauge_in_file.c_str());
    }
    else {
        fprintf(stdout, "gauge_in_file is not configured\n");
    }
    // GAUGE_OUT_FILE name
    if (gauge_out_file_configured) {
        fprintf(stdout, "gauge_out_file = %s\n", gauge_out_file.c_str());
    }
    else {
        fprintf(stdout, "gauge_out_file is not configured\n");
    }
    // FERMION_IN_FILE name
    if (fermion_in_file_configured) {
        fprintf(stdout, "fermion_in_file = %s\n", fermion_in_file.c_str());
    }
    else {
        fprintf(stdout, "fermion_in_file is not configured\n");
    }
    // FERMION_OUT_FILE name
    if (fermion_out_file_configured) {
        fprintf(stdout, "fermion_out_file = %s\n", fermion_out_file.c_str());
    }
    else {
        fprintf(stdout, "fermion_out_file is not configured\n");
    }
}

static void print_usage(const char* arg) {
    fprintf(stderr, "Usage: %s --mpi mpi_str --lattice latt_size --Nc c --mIn m [--gauge_in file1 ] [--gauge_out file2] [--fermion_in file3] [--fermion_out file4] \n", arg);
    fprintf(stderr, "          --mpi  x.y.z.t         how many mpi in each direction, example 1.1.1.1\n");
    fprintf(stderr, "          --lattice  x.y.z.t     how many lattice in each direction, example 4.4.4.4\n");
    fprintf(stderr, "          --color  c             how many color, example 3\n");
    fprintf(stderr, "          --mIn m                how many mIn, example 12\n");
    fprintf(stderr, " optional --gauge_in    filename       file name\n");
    fprintf(stderr, " optional --gauge_out   filename       file name\n");
    fprintf(stderr, " optional --fermion_in  filename       file name\n");
    fprintf(stderr, " optional --fermion_out filename       file name\n");
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

// QcuHeader get_lattice_config(int argc, char *argv[], std::string& file, MPI_Desc& mpi_desc) {
void get_lattice_config(
    int argc,                           // in 
    char *argv[],                       // in
    TerminalConfig &config
)
{
    // 默认为空文件名
    config.gauge_in_file    = "";
    config.gauge_out_file   = "";
    config.fermion_in_file  = "";
    config.fermion_out_file = "";

    bool lattice_setted = false;
    // mpi, Nc, mIn必须设置
    bool mpi_setted = false;
    bool Nc_setted = false;
    bool mIn_setted = false;
    int  Nc = 0;
    int  mInput = 0;

    int opt;                // getopt_long() 的返回值
    int digit_optind = 0;   // 设置短参数类型及是否需要参数

    int option_index = 0;
    const char *optstring = "m:l:c:i:";

    static struct option long_options[] = {
        {"mpi",         required_argument, NULL, 'm'},
        {"lattice",     required_argument, NULL, 'l'},
        {"color",       required_argument, NULL, 'c'},
        {"mIn",         required_argument, NULL, 'i'},
        {"gauge_in",    required_argument, NULL,'g'},
        {"gauge_out",   required_argument, NULL,'G'},
        {"fermion_in",  required_argument, NULL,'f'},
        {"fermion_out", required_argument, NULL,'F'},
        {"help",        no_argument,       NULL, 'h'},
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
                config.mpi_desc.data[X_DIM] = mpiDesc[X_DIM];
                config.mpi_desc.data[Y_DIM] = mpiDesc[Y_DIM];
                config.mpi_desc.data[Z_DIM] = mpiDesc[Z_DIM];
                config.mpi_desc.data[T_DIM] = mpiDesc[T_DIM];
                config.mpi_configured = 1;
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
                config.lattice_desc.data[X_DIM] = lattice[X_DIM];
                config.lattice_desc.data[Y_DIM] = lattice[Y_DIM];
                config.lattice_desc.data[Z_DIM] = lattice[Z_DIM];
                config.lattice_desc.data[T_DIM] = lattice[T_DIM];
                config.lattice_configured = 1;
                lattice_setted = true;
            }
            break;
        case 'c': // --Nc or -c
            {
                Nc = stoi(std::string(optarg));
                config.Nc = Nc;
                Nc_setted = true;
            }
            break;
        case 'i': // --mIn or -i
            {
                mInput = stoi(std::string(optarg));
                config.mInput = mInput;
                mIn_setted = true;
            }
            break;
        case 'g': // --gauge_in or -g
            {
                config.gauge_in_file = std::string(optarg);
                config.gauge_in_file_configured = 1;
            }
            break;
        case 'G': // --gauge_out or -G
            {
                config.gauge_out_file = std::string(optarg);
                config.gauge_out_file_configured = 1;
            }
            break;
        case 'f': // --file or -f
            {
                config.fermion_in_file = std::string(optarg);
                config.fermion_in_file_configured = 1;
            }
            break;
        case 'F': // --file or -F
            {
                config.fermion_out_file = std::string(optarg);
                config.fermion_out_file_configured = 1;
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
}

