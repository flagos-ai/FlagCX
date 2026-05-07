#include "tools.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <libgen.h>

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
      // {"op", required_argument, 0, 'o'},
      // {"datatype", required_argument, 0, 'd'},
      {"help", no_argument, 0, 'h'},
      {}};

  while (1) {
    int c;
    c = getopt_long(argc, argv, "b:e:f:w:n:p:r:s:R:h", longOpts, &longIndex);

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
               "[-h\n",
               basename(argv[0]));
        printf("Use default values with -b 1M -e 1G -f 2 -w 5 -n 20 -p 0 -r 0 "
               "-s 0 -R 0\n");
        break;
    }
  }
}
