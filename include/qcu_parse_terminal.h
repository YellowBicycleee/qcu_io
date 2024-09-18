#pragma once
#include "qcu_mpi_desc.h"
#include "lattice_desc.h"
#include "lqcd_read_write.h"

QcuHeader get_lattice_config(int argc, char *argv[], std::string& file, MPI_Desc& mpi_desc);
