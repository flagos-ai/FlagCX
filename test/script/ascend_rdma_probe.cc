// Independent Ascend NPU-memory registration and mlx5 loopback data probe.
// This deliberately uses plain verbs, outside FlagCX's MR and proxy paths.
#include <acl/acl.h>
#include <infiniband/verbs.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const size_t kBytes = 4096;

static bool compare_bytes(const char *stage, const uint8_t *actual,
                          const uint8_t *expected) {
  for (size_t i = 0; i < kBytes; ++i) {
    if (actual[i] != expected[i]) {
      fprintf(stderr,
              "ASCEND_RDMA_PROBE %s=failed mismatch_offset=%zu expected=0x%02x "
              "actual=0x%02x\n",
              stage, i, (unsigned)expected[i], (unsigned)actual[i]);
      return false;
    }
  }
  printf("ASCEND_RDMA_PROBE %s=passed\n", stage);
  return true;
}

static bool copy_acl(const char *stage, void *dst, const void *src,
                     aclrtMemcpyKind kind) {
  aclError status = aclrtMemcpy(dst, kBytes, src, kBytes, kind);
  if (status != ACL_SUCCESS) {
    fprintf(stderr, "ASCEND_RDMA_PROBE %s=failed acl=%d\n", stage, (int)status);
    return false;
  }
  return true;
}

static bool choose_gid(ibv_context *ctx, uint8_t port, int count, ibv_gid *gid,
                       int *gid_index) {
  const char *requested = getenv("FLAGCX_CI_IB_GID_INDEX");
  if (requested != NULL) {
    char *end = NULL;
    long index = strtol(requested, &end, 10);
    if (*requested == '\0' || *end != '\0' || index < 0 || index >= count ||
        ibv_query_gid(ctx, port, (int)index, gid) != 0)
      return false;
    *gid_index = (int)index;
    return true;
  }
  // Index 3 is the IPv4 RoCE-v2 GID on the current Ascend CI host.
  for (int attempt = -1; attempt < count; ++attempt) {
    int index = attempt == -1 ? 3 : attempt;
    if (index >= count || (attempt >= 0 && index == 3))
      continue;
    ibv_gid candidate = {};
    if (ibv_query_gid(ctx, port, index, &candidate) != 0)
      continue;
    const uint8_t zero[16] = {};
    if (memcmp(candidate.raw, zero, sizeof(zero)) == 0)
      continue;
    *gid = candidate;
    *gid_index = index;
    return true;
  }
  return false;
}

static bool set_qp_states(ibv_qp *qp, uint8_t port, ibv_qp *peer,
                          const ibv_port_attr *port_attr, const ibv_gid *gid,
                          int gid_index) {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                        IBV_QP_ACCESS_FLAGS) != 0)
    return false;

  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr->active_mtu;
  attr.dest_qp_num = peer->qp_num;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.port_num = port;
  if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = *gid;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 1;
  } else {
    attr.ah_attr.dlid = port_attr->lid;
  }
  if (ibv_modify_qp(qp, &attr,
                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
    return false;

  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  return ibv_modify_qp(qp, &attr,
                       IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                           IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_MAX_QP_RD_ATOMIC) == 0;
}

static bool transfer(const char *stage, ibv_qp *qp, ibv_cq *cq,
                     ibv_mr *local_mr, ibv_mr *remote_mr,
                     ibv_wr_opcode operation) {
  static uint64_t next_wr_id = 1;
  const uint64_t wr_id = next_wr_id++;
  ibv_sge sge = {};
  sge.addr = (uintptr_t)local_mr->addr;
  sge.length = kBytes;
  sge.lkey = local_mr->lkey;
  ibv_send_wr wr = {};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = operation;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uintptr_t)remote_mr->addr;
  wr.wr.rdma.rkey = remote_mr->rkey;
  ibv_send_wr *bad = NULL;
  if (ibv_post_send(qp, &wr, &bad) != 0) {
    fprintf(stderr,
            "ASCEND_RDMA_PROBE %s post_send=failed operation=%d "
            "errno=%d (%s)\n",
            stage, (int)operation, errno, strerror(errno));
    return false;
  }
  struct timespec start = {};
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (;;) {
    ibv_wc wc = {};
    int count = ibv_poll_cq(cq, 1, &wc);
    if (count > 0) {
      if (wc.wr_id != wr_id) {
        fprintf(stderr,
                "ASCEND_RDMA_PROBE %s stale_completion wr_id=%llu "
                "status=%s vendor_err=%u\n",
                stage, (unsigned long long)wc.wr_id,
                ibv_wc_status_str(wc.status), wc.vendor_err);
        continue;
      }
      printf("ASCEND_RDMA_PROBE %s completion=%s vendor_err=%u\n", stage,
             ibv_wc_status_str(wc.status), wc.vendor_err);
      return wc.status == IBV_WC_SUCCESS;
    }
    if (count < 0) {
      fprintf(stderr, "ASCEND_RDMA_PROBE %s poll_cq=failed count=%d\n", stage,
              count);
      return false;
    }
    struct timespec now = {};
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (now.tv_sec - start.tv_sec >= 10) {
      fprintf(stderr, "ASCEND_RDMA_PROBE %s completion=timeout\n", stage);
      return false;
    }
    usleep(1000);
  }
}

int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  int result = 1;
  bool all_passed = true;
  bool acl_ready = false;
  bool device_ready = false;
  void *allocation = NULL;
  void *host = NULL;
  void *host_target = NULL;
  ibv_device **devices = NULL;
  ibv_context *ctx = NULL;
  ibv_pd *pd = NULL;
  ibv_cq *cq = NULL;
  ibv_qp *source = NULL;
  ibv_qp *target = NULL;
  ibv_mr *host_mr = NULL;
  ibv_mr *host_target_mr = NULL;
  ibv_mr *npu_mr = NULL;
  uint8_t port = 0;
  ibv_port_attr port_attr = {};
  ibv_gid gid = {};
  int gid_index = -1;
  uint8_t expected[kBytes] = {};
  uint8_t observed[kBytes] = {};
  uint8_t initial[kBytes] = {};

  size_t alignment = kBytes;
  const char *requested_alignment = getenv("FLAGCX_CI_RDMA_ALIGNMENT");
  if (requested_alignment != NULL) {
    if (strcmp(requested_alignment, "4096") == 0)
      alignment = 4096;
    else if (strcmp(requested_alignment, "2097152") == 0)
      alignment = 2097152;
    else {
      fprintf(stderr, "ASCEND_RDMA_PROBE invalid alignment=%s\n",
              requested_alignment);
      printf("ASCEND_RDMA_PROBE result=failed\n");
      return 2;
    }
  }

  do {
    aclError acl_result = aclInit(NULL);
    if (acl_result != ACL_SUCCESS) {
      fprintf(stderr, "ASCEND_RDMA_PROBE aclInit=%d\n", (int)acl_result);
      break;
    }
    acl_ready = true;
    acl_result = aclrtSetDevice(0);
    if (acl_result != ACL_SUCCESS) {
      fprintf(stderr, "ASCEND_RDMA_PROBE aclrtSetDevice=%d\n", (int)acl_result);
      break;
    }
    device_ready = true;
    const size_t allocation_size = alignment + kBytes;
    acl_result =
        aclrtMalloc(&allocation, allocation_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_result != ACL_SUCCESS || allocation == NULL) {
      fprintf(stderr, "ASCEND_RDMA_PROBE aclrtMalloc=%d\n", (int)acl_result);
      break;
    }
    const uintptr_t npu_addr =
        ((uintptr_t)allocation + alignment - 1) & ~(uintptr_t)(alignment - 1);
    printf("ASCEND_RDMA_PROBE npu_allocation=%p allocation_size=%zu "
           "alignment=%zu aligned_range=%p size=%zu\n",
           allocation, allocation_size, alignment, (void *)npu_addr, kBytes);
    if (posix_memalign(&host, kBytes, kBytes) != 0 ||
        posix_memalign(&host_target, kBytes, kBytes) != 0) {
      fprintf(stderr, "ASCEND_RDMA_PROBE host_allocation=failed\n");
      break;
    }

    for (size_t i = 0; i < kBytes; ++i)
      expected[i] = (uint8_t)(i * 37 + 11);
    if (!copy_acl("acl_baseline_h2d", (void *)npu_addr, expected,
                  ACL_MEMCPY_HOST_TO_DEVICE) ||
        !copy_acl("acl_baseline_d2h", observed, (void *)npu_addr,
                  ACL_MEMCPY_DEVICE_TO_HOST)) {
      all_passed = false;
    } else if (!compare_bytes("acl_baseline", observed, expected)) {
      all_passed = false;
    }

    int count = 0;
    devices = ibv_get_device_list(&count);
    if (devices == NULL || count == 0) {
      fprintf(stderr, "ASCEND_RDMA_PROBE no verbs devices\n");
      break;
    }
    const char *requested_hca = getenv("FLAGCX_CI_IB_HCA");
    for (int i = 0; i < count && ctx == NULL; ++i) {
      const char *name = ibv_get_device_name(devices[i]);
      if ((requested_hca != NULL && strcmp(name, requested_hca) != 0) ||
          (requested_hca == NULL && strncmp(name, "mlx5_", 5) != 0))
        continue;
      ibv_context *candidate = ibv_open_device(devices[i]);
      if (candidate == NULL)
        continue;
      ibv_device_attr device_attr = {};
      if (ibv_query_device(candidate, &device_attr) == 0) {
        for (int p = 1; p <= device_attr.phys_port_cnt; ++p) {
          ibv_port_attr attr = {};
          if (ibv_query_port(candidate, p, &attr) == 0 &&
              attr.state == IBV_PORT_ACTIVE &&
              (attr.link_layer == IBV_LINK_LAYER_ETHERNET ||
               attr.link_layer == IBV_LINK_LAYER_INFINIBAND)) {
            ctx = candidate;
            port = (uint8_t)p;
            port_attr = attr;
            break;
          }
        }
      }
      if (ctx == NULL)
        ibv_close_device(candidate);
    }
    if (ctx == NULL) {
      fprintf(stderr, "ASCEND_RDMA_PROBE no active verbs port\n");
      break;
    }
    printf("ASCEND_RDMA_PROBE hca=%s port=%u link_layer=%d\n",
           ibv_get_device_name(ctx->device), port, port_attr.link_layer);
    if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
        !choose_gid(ctx, port, port_attr.gid_tbl_len, &gid, &gid_index)) {
      fprintf(stderr, "ASCEND_RDMA_PROBE no usable RoCE GID\n");
      break;
    }
    if (gid_index >= 0)
      printf("ASCEND_RDMA_PROBE gid_index=%d\n", gid_index);
    pd = ibv_alloc_pd(ctx);
    if (pd == NULL)
      break;

    host_mr = ibv_reg_mr(pd, host, kBytes, IBV_ACCESS_LOCAL_WRITE);
    if (host_mr == NULL) {
      fprintf(stderr, "ASCEND_RDMA_PROBE host_mr=failed errno=%d (%s)\n", errno,
              strerror(errno));
      break;
    }
    printf("ASCEND_RDMA_PROBE host_mr=passed\n");
    host_target_mr =
        ibv_reg_mr(pd, host_target, kBytes,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);
    if (host_target_mr == NULL) {
      fprintf(stderr, "ASCEND_RDMA_PROBE host_target_mr=failed errno=%d (%s)\n",
              errno, strerror(errno));
      break;
    }
    printf("ASCEND_RDMA_PROBE host_target_mr=passed\n");
    const int npu_access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
    errno = 0;
    npu_mr = ibv_reg_mr(pd, (void *)npu_addr, kBytes, npu_access);
    if (npu_mr == NULL) {
      fprintf(stderr, "ASCEND_RDMA_PROBE npu_mr=failed errno=%d (%s)\n", errno,
              strerror(errno));
      break;
    }
    printf("ASCEND_RDMA_PROBE npu_mr=passed\n");

    typedef ibv_mr *(*RegMrIova2)(ibv_pd *, void *, size_t, uint64_t, int);
    RegMrIova2 reg_iova2 = (RegMrIova2)dlsym(RTLD_DEFAULT, "ibv_reg_mr_iova2");
    if (reg_iova2 != NULL) {
      errno = 0;
      ibv_mr *relaxed_mr = reg_iova2(pd, (void *)npu_addr, kBytes, npu_addr,
                                     npu_access | IBV_ACCESS_REMOTE_ATOMIC |
                                         IBV_ACCESS_RELAXED_ORDERING);
      if (relaxed_mr != NULL) {
        printf("ASCEND_RDMA_PROBE flagcx_like_iova2_mr=passed\n");
        ibv_dereg_mr(relaxed_mr);
      } else {
        printf("ASCEND_RDMA_PROBE flagcx_like_iova2_mr=failed errno=%d (%s)\n",
               errno, strerror(errno));
        all_passed = false;
      }
    } else {
      printf("ASCEND_RDMA_PROBE flagcx_like_iova2_mr=unavailable\n");
    }

    cq = ibv_create_cq(ctx, 8, NULL, NULL, 0);
    if (cq == NULL)
      break;
    ibv_qp_init_attr qp_init = {};
    qp_init.send_cq = cq;
    qp_init.recv_cq = cq;
    qp_init.cap.max_send_wr = 4;
    qp_init.cap.max_recv_wr = 1;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    qp_init.qp_type = IBV_QPT_RC;
    source = ibv_create_qp(pd, &qp_init);
    target = ibv_create_qp(pd, &qp_init);
    if (source == NULL || target == NULL ||
        !set_qp_states(source, port, target, &port_attr, &gid, gid_index) ||
        !set_qp_states(target, port, source, &port_attr, &gid, gid_index)) {
      fprintf(stderr, "ASCEND_RDMA_PROBE qp_setup=failed errno=%d (%s)\n",
              errno, strerror(errno));
      break;
    }
    printf("ASCEND_RDMA_PROBE qp_setup=passed\n");

    memcpy(host, expected, kBytes);
    memset(host_target, 0, kBytes);
    if (!transfer("host_rdma_write", source, cq, host_mr, host_target_mr,
                  IBV_WR_RDMA_WRITE)) {
      fprintf(stderr, "ASCEND_RDMA_PROBE host_rdma_write=failed\n");
      all_passed = false;
    } else if (!compare_bytes("host_rdma_write", (uint8_t *)host_target,
                              expected)) {
      all_passed = false;
    }
    memset(host, 0, kBytes);
    if (!transfer("host_rdma_read", source, cq, host_mr, host_target_mr,
                  IBV_WR_RDMA_READ)) {
      fprintf(stderr, "ASCEND_RDMA_PROBE host_rdma_read=failed\n");
      all_passed = false;
    } else if (!compare_bytes("host_rdma_read", (uint8_t *)host, expected)) {
      all_passed = false;
    }

    memcpy(host, expected, kBytes);

    memset(initial, 0, kBytes);
    bool write_ready = copy_acl("npu_write_prepare", (void *)npu_addr, initial,
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (write_ready)
      write_ready = copy_acl("npu_write_baseline_d2h", observed,
                             (void *)npu_addr, ACL_MEMCPY_DEVICE_TO_HOST);
    if (write_ready)
      write_ready = compare_bytes("npu_write_baseline", observed, initial);
    if (!write_ready) {
      printf("ASCEND_RDMA_PROBE npu_rdma_write=skipped acl_baseline_failed\n");
      all_passed = false;
    } else if (!transfer("npu_rdma_write", source, cq, host_mr, npu_mr,
                         IBV_WR_RDMA_WRITE)) {
      fprintf(stderr, "ASCEND_RDMA_PROBE npu_rdma_write=failed\n");
      all_passed = false;
    } else {
      // Read the same MR with verbs before ACL can rewrite the NPU buffer.
      // This separates an RNIC write failure from an ACL visibility issue.
      memset(host, 0, kBytes);
      bool rdma_view_passed = transfer("npu_write_rdma_readback", source, cq,
                                       host_mr, npu_mr, IBV_WR_RDMA_READ);
      if (rdma_view_passed)
        rdma_view_passed =
            compare_bytes("npu_write_rdma_readback", (uint8_t *)host, expected);
      else
        fprintf(stderr, "ASCEND_RDMA_PROBE npu_write_rdma_readback=failed\n");

      bool acl_view_passed =
          copy_acl("npu_write_acl_d2h", observed, (void *)npu_addr,
                   ACL_MEMCPY_DEVICE_TO_HOST);
      if (acl_view_passed)
        acl_view_passed =
            compare_bytes("npu_write_acl_readback", observed, expected);
      if (!rdma_view_passed || !acl_view_passed)
        all_passed = false;
      printf("ASCEND_RDMA_PROBE npu_rdma_write=%s\n",
             rdma_view_passed && acl_view_passed ? "passed" : "failed");
    }

    for (size_t i = 0; i < kBytes; ++i)
      expected[i] = (uint8_t)(i * 19 + 7);
    bool read_ready = copy_acl("npu_read_prepare", (void *)npu_addr, expected,
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (read_ready)
      read_ready = copy_acl("npu_read_baseline_d2h", observed, (void *)npu_addr,
                            ACL_MEMCPY_DEVICE_TO_HOST);
    if (read_ready)
      read_ready = compare_bytes("npu_read_baseline", observed, expected);
    memset(host, 0, kBytes);
    if (!read_ready) {
      printf("ASCEND_RDMA_PROBE npu_rdma_read=skipped acl_baseline_failed\n");
      all_passed = false;
    } else if (!transfer("npu_rdma_read", source, cq, host_mr, npu_mr,
                         IBV_WR_RDMA_READ)) {
      fprintf(stderr, "ASCEND_RDMA_PROBE npu_rdma_read=failed\n");
      all_passed = false;
    } else if (!compare_bytes("npu_rdma_read", (uint8_t *)host, expected)) {
      all_passed = false;
    }
    if (all_passed)
      result = 0;
  } while (0);

  if (target != NULL)
    ibv_destroy_qp(target);
  if (source != NULL)
    ibv_destroy_qp(source);
  if (cq != NULL)
    ibv_destroy_cq(cq);
  if (npu_mr != NULL)
    ibv_dereg_mr(npu_mr);
  if (host_mr != NULL)
    ibv_dereg_mr(host_mr);
  if (host_target_mr != NULL)
    ibv_dereg_mr(host_target_mr);
  if (pd != NULL)
    ibv_dealloc_pd(pd);
  if (ctx != NULL)
    ibv_close_device(ctx);
  if (devices != NULL)
    ibv_free_device_list(devices);
  free(host);
  free(host_target);
  if (allocation != NULL)
    aclrtFree(allocation);
  if (device_ready)
    aclrtResetDevice(0);
  if (acl_ready)
    aclFinalize();
  printf("ASCEND_RDMA_PROBE result=%s\n", result == 0 ? "passed" : "failed");
  return result;
}
