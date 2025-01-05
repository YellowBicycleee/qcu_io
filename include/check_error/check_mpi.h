#pragma once

#define CHECK_MPI(cmd)                                                   \
    do {                                                                 \
        int err = cmd;                                                   \
        if (err != MPI_SUCCESS) {                                        \
            char error_string[MPI_MAX_ERROR_STRING];                     \
            int error_string_length;                                     \
            MPI_Error_string(err, error_string, &error_string_length);   \
            fprintf(stderr, "MPI error in %s at line %d: %d - %s\n",     \
                    __FILE__, __LINE__, err, error_string);              \
            MPI_Abort(MPI_COMM_WORLD, err);                              \
        }                                                                \
    } while (0)



