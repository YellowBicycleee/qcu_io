#pragma once

#define CHECK_MPI(cmd)                               \
    do {                                             \
        int err = cmd;                               \
        if (err != MPI_SUCCESS) {                    \
            fprintf(stderr, "MPI error: %d\n", err); \
            exit(1);                                 \
        }                                            \
    } while (0)


