## Tests

Tests for FlagCX are maintained in `test/perf`.

```shell
cd test/perf
make [USE_NVIDIA | USE_ILUVATAR_COREX | USE_CAMBRICON | USE_METAX | USE_MUSA | USE_KUNLUNXIN | USE_DU | USE_ASCEND | USE_TSM]=1
mpirun --allow-run-as-root -np 8 ./test_allreduce -b 128K -e 4G -f 2
```

Note that the default MPI install path is set to `/usr/local/mpi`, you may specify the MPI path with:

```shell
make MPI_HOME=<MPI path>
```

All tests support the same set of arguments:

- Sizes to scan

  * `-b <min>` minimum size in bytes to start with. Default: 1M.
  * `-e <max>` maximum size in bytes to end at. Default: 1G.
  * `-f <increment factor>` multiplication factor between sizes. Default: 2.

- Performance

  * `-w, <warmup iterations >` number of warmup iterations (not timed). Default: 5.
  * `-n, <iterations >` number of iterations. Default: 20.

- Test operation

  * `-R, <0/1>` enable local buffer registration on send/recv buffers. Default: 0.
  * `-s, <OCT/DEC/HEX>` specify MPI communication split mode. Default: 0

- Utils

  * `-p, <0/1>` print buffer info. Default: 0.
  * `-h` print help message. Default: disabled.

