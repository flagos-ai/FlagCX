/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#include "probe.h"

#include <dirent.h>

std::vector<ProbePort> probePorts() {
  std::vector<ProbePort> result;
  // Discovery must not open unrelated HCAs: a broken provider/device must
  // not prevent allocation controls or probes on another HCA from running.
  DIR *devices = opendir("/sys/class/infiniband");
  if (!devices)
    return result;
  while (auto *device = readdir(devices)) {
    const std::string name = device->d_name;
    if (name.empty() || name[0] == '.')
      continue;
    const std::string root = "/sys/class/infiniband/" + name + "/ports/";
    DIR *entries = opendir(root.c_str());
    if (!entries)
      continue;
    while (auto *entry = readdir(entries)) {
      const int port = std::atoi(entry->d_name);
      if (port <= 0)
        continue;
      std::ifstream stateFile(root + entry->d_name + "/state");
      int state = 0;
      stateFile >> state;
      // Linux sysfs port state 4 means ACTIVE; no verbs ABI is needed here.
      if (state == 4)
        result.push_back({name, port});
    }
    closedir(entries);
  }
  closedir(devices);
  std::sort(result.begin(), result.end(),
            [](const ProbePort &a, const ProbePort &b) {
              return a.hca == b.hca ? a.port < b.port : a.hca < b.hca;
            });
  return result;
}

Json inventory() {
  Json result = {{"event", "inventory"}, {"ports", Json::array()}};
  int gpus = 0, runtime = 0, driver = 0;
  result["gpu_count_code"] = int(mcGetDeviceCount(&gpus));
  result["gpu_count"] = gpus;
  result["runtime_code"] = int(mcRuntimeGetVersion(&runtime));
  result["runtime_version"] = runtime;
  result["driver_code"] = int(mcDriverGetVersion(&driver));
  result["driver_version"] = driver;
  for (const auto &port : probePorts())
    result["ports"].push_back(port.hca + ":" + std::to_string(port.port));
  return result;
}
