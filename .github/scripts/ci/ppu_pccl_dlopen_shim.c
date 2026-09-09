#define _GNU_SOURCE

#include <dlfcn.h>
#include <string.h>

typedef void *(*dlopen_fn)(const char *, int);

void *dlopen(const char *filename, int flags) {
  static dlopen_fn real_dlopen;

  if (real_dlopen == NULL) {
    *(void **)(&real_dlopen) = dlsym(RTLD_NEXT, "dlopen");
    if (real_dlopen == NULL)
      return NULL;
  }

#ifdef RTLD_DEEPBIND
  if (filename != NULL) {
    const char *basename = strrchr(filename, '/');
    basename = basename == NULL ? filename : basename + 1;
    if (strcmp(basename, "libpccl.so") == 0 ||
        strncmp(basename, "libpccl.so.", sizeof("libpccl.so.") - 1) == 0)
      flags |= RTLD_DEEPBIND;
  }
#endif

  return real_dlopen(filename, flags);
}
