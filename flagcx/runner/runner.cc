/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "runner.h"

struct flagcxRunner *flagcxRunners[NRUNNERS] = {&homoRunner, &hostRunner,
                                                &hybridRunner, &uniRunner};