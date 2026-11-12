#include "tools.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <libgen.h>

// Datatype tables
const flagcxDataType_t test_types[] = {
    flagcxInt8,   flagcxUint8,   flagcxInt32,   flagcxUint32,  flagcxInt64,
    flagcxUint64, flagcxFloat16, flagcxFloat32, flagcxFloat64, flagcxBfloat16};
const char *test_typenames[] = {"int8",   "uint8",   "int32", "uint32",
                                "int64",  "uint64",  "half",  "float",
                                "double", "bfloat16"};
int test_typenum = 10;

// Reduction op tables
// Note: flagcxAvg is excluded from the default "all" sweep (test_opnum=4)
// because it is not supported by all adaptors (MPI, Gloo). Users can still
// explicitly select it with --op avg on supported backends.
const flagcxRedOp_t test_ops[] = {flagcxSum, flagcxProd, flagcxMax, flagcxMin,
                                  flagcxAvg};
const char *test_opnames[] = {"sum", "prod", "max", "min", "avg"};
int test_opnum = 4; // only sweep sum/prod/max/min in "all" mode

int flagcxStringToType(const char *str) {
  for (int t = 0; t < test_typenum; t++) {
    if (strcmp(str, test_typenames[t]) == 0)
      return t;
  }
  if (strcmp(str, "all") == 0)
    return -1;
  printf("invalid datatype '%s', defaulting to float\n", str);
  return flagcxFloat;
}

int flagcxStringToOp(const char *str) {
  // Search all ops including avg (which is beyond test_opnum)
  static const int totalOps = sizeof(test_ops) / sizeof(test_ops[0]);
  for (int o = 0; o < totalOps; o++) {
    if (strcmp(str, test_opnames[o]) == 0)
      return o;
  }
  if (strcmp(str, "all") == 0)
    return -1;
  printf("invalid op '%s', defaulting to sum\n", str);
  return flagcxSum;
}

void initMpiEnv(int argc, char **argv, int &worldRank, int &worldSize,
                int &proc, int &totalProcs, int &color, MPI_Comm &splitComm,
                uint64_t splitMask) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  printf("I am %d of %d\n", worldRank, worldSize);

  color = worldRank & splitMask;
  MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &splitComm);
  MPI_Comm_size(splitComm, &totalProcs);
  MPI_Comm_rank(splitComm, &proc);
  printf("I am %d of %d in group %d\n", proc, totalProcs, color);
}

std::uint64_t now() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

timer::timer() { start = now(); }

double timer::elapsed() const {
  std::uint64_t end = now();
  return 1.e-9 * (end - start);
}

double timer::reset() {
  std::uint64_t end = now();
  double ans = 1.e-9 * (end - start);
  start = end;
  return ans;
}

double parsesize(const char *value) {
  long long int units;
  double size;
  char sizeLit;

  int count = sscanf(value, "%lf %1s", &size, &sizeLit);

  switch (count) {
    case 2:
      switch (sizeLit) {
        case 'G':
        case 'g':
          units = 1024 * 1024 * 1024;
          break;
        case 'M':
        case 'm':
          units = 1024 * 1024;
          break;
        case 'K':
        case 'k':
          units = 1024;
          break;
        default:
          return -1.0;
      };
      break;
    case 1:
      units = 1;
      break;
    default:
      return -1.0;
  }

  return size * units;
}

parser::parser(int argc, char **argv) {
  minBytes = 1ULL * 1024 * 1024;
  maxBytes = 1ULL * 1024 * 1024 * 1024;
  stepFactor = 2;
  warmupIters = 5;
  testIters = 20;
  printBuffer = 0;
  root = -1;
  splitMask = 0;
  localRegister = 0;
  verifyData = 1;
  datatype = flagcxFloat;
  op = flagcxSum;

  double parsedValue;
  int longIndex;
  static struct option longOpts[] = {
      {"minbytes", required_argument, 0, 'b'},
      {"maxbytes", required_argument, 0, 'e'},
      {"stepfactor", required_argument, 0, 'f'},
      {"warmup_iters", required_argument, 0, 'w'},
      {"iters", required_argument, 0, 'n'},
      {"print_buffer", required_argument, 0, 'p'},
      {"root", required_argument, 0, 'r'},
      {"split_mask", required_argument, 0, 's'},
      {"local_register", required_argument, 0, 'R'},
      {"verify", required_argument, 0, 'v'},
      {"op", required_argument, 0, 'o'},
      {"datatype", required_argument, 0, 'd'},
      {"help", no_argument, 0, 'h'},
      {}};

  while (1) {
    int c;
    c = getopt_long(argc, argv, "b:e:f:w:n:p:r:s:R:v:o:d:h", longOpts,
                    &longIndex);

    if (c == -1)
      break;

    switch (c) {
      case 'b':
        parsedValue = parsesize(optarg);
        if (parsedValue < 0) {
          fprintf(stderr, "Invalid minbytes value\n");
          exit(1);
        }
        minBytes = (size_t)parsedValue;
        break;
      case 'e':
        parsedValue = parsesize(optarg);
        if (parsedValue < 0) {
          fprintf(stderr, "Invalid maxbytes value\n");
          exit(1);
        }
        maxBytes = (size_t)parsedValue;
        break;
      case 'f':
        stepFactor = (int)strtol(optarg, NULL, 0);
        if (stepFactor < 1) {
          fprintf(stderr, "Invalid stepfactor value\n");
          exit(1);
        }
        break;
      case 'w':
        warmupIters = (int)strtol(optarg, NULL, 0);
        if (warmupIters < 0) {
          fprintf(stderr, "Invalid warmupIters value\n");
          exit(1);
        }
        break;
      case 'n':
        testIters = (int)strtol(optarg, NULL, 0);
        if (testIters < 0) {
          fprintf(stderr, "Invalid testIters value\n");
          exit(1);
        }
        break;
      case 'p':
        printBuffer = (int)strtol(optarg, NULL, 0);
        if (printBuffer != 0 && printBuffer != 1) {
          fprintf(stderr, "Invalid printBuffer value\n");
          exit(1);
        }
        break;
      case 'r':
        root = (int)strtol(optarg, NULL, 0);
        if (root < 0) {
          fprintf(stderr, "Invalid root value\n");
          exit(1);
        }
        break;
      case 's':
        splitMask = strtoul(optarg, NULL, 0);
        break;
      case 'R':
        localRegister = (int)strtol(optarg, NULL, 0);
        if (localRegister < 0 || localRegister > 2) {
          fprintf(stderr,
                  "Invalid local register value: %d (must be 0, 1, or 2)\n",
                  localRegister);
          exit(1);
        }
        break;
      case 'v':
        if (strcmp(optarg, "0") != 0 && strcmp(optarg, "1") != 0) {
          fprintf(stderr, "Invalid verification value: %s (must be 0 or 1)\n",
                  optarg);
          exit(1);
        }
        verifyData = optarg[0] - '0';
        break;
      case 'd':
        datatype = flagcxStringToType(optarg);
        break;
      case 'o':
        op = flagcxStringToOp(optarg);
        break;
      case 'h':
      default:
        if (c != 'h')
          printf("Invalid argument '%c'\n", c);
        printf("Usage: %s \n\t"
               "[-b <minbytes K/M/G>] \n\t"
               "[-e <maxbytes K/M/G>] \n\t"
               "[-f <stepfactor>] \n\t"
               "[-w <warmupiters>] \n\t"
               "[-n <iters>] \n\t"
               "[-p <printbuffer 0/1>] \n\t"
               "[-r <root>] \n\t"
               "[-s <splitmask OCT/DEC/HEX>] \n\t"
               "[-R <localregister 0/1/2>] \n\t"
               "[-v <verify 0/1>] \n\t"
               "[-d <datatype/all>] \n\t"
               "[-o <op/all>] \n\t"
               "[-h\n",
               basename(argv[0]));
        printf("Use default values with -b 1M -e 1G -f 2 -w 5 -n 20 -p 0 -r 0 "
               "-s 0 -R 0 -v 1 -d float -o sum\n");
        break;
    }
  }
}
