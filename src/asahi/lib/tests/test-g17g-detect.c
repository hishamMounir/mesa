/*
 * Copyright 2026 Gravity Linux Contributors
 * SPDX-License-Identifier: MIT
 *
 * agx_open_device itself, with the kernel replies faked: a G17G (Apple M5)
 * report is recognized and refused before any VM setup, while G17P still
 * proceeds to it. No GPU or DRM device is needed.
 */
#include "agx_device.h"
#include "agx_native_mesh.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static jmp_buf reached_vm_setup;
static char variant;
static unsigned other_ioctls;

drmVersionPtr __wrap_drmGetVersion(int fd);
drmVersionPtr
__wrap_drmGetVersion(int fd)
{
   static drmVersion version = {.name = "asahi"};
   CHECK(fd == 101);
   return &version;
}

void __wrap_drmFreeVersion(drmVersionPtr version);
void
__wrap_drmFreeVersion(drmVersionPtr version)
{
   (void)version;
}

int __wrap_drmIoctl(int fd, unsigned long request, void *arg);
int
__wrap_drmIoctl(int fd, unsigned long request, void *arg)
{
   CHECK(fd == 101);
   if (request != DRM_IOCTL_ASAHI_GET_PARAMS) {
      other_ioctls++;
      return -1;
   }
   struct drm_asahi_get_params *get = arg;
   struct drm_asahi_params_global *p = (void *)(uintptr_t)get->pointer;
   *p = (struct drm_asahi_params_global){
      .gpu_generation = 17,
      .gpu_variant = variant,
      .vm_start = 0,
      .vm_end = 1ull << 42,
      .vm_kernel_min_size = 1ull << 26,
      .command_timestamp_frequency_hz = 24000000,
   };
   return 0;
}

/* The first call after the address-space layout; reaching it means the open
 * went on toward VM setup. */
struct agx_native_mesh_support
__wrap_agx_native_mesh_probe(int fd, const char *classes);
struct agx_native_mesh_support
__wrap_agx_native_mesh_probe(int fd, const char *classes)
{
   (void)fd;
   (void)classes;
   longjmp(reached_vm_setup, 1);
}

static bool
open_as(char v, struct agx_device *dev, bool *reached)
{
   memset(dev, 0, sizeof(*dev));
   dev->fd = 101;
   variant = v;
   other_ioctls = 0;
   *reached = false;
   if (setjmp(reached_vm_setup)) {
      *reached = true;
      return true;
   }
   return agx_open_device(NULL, dev);
}

int
main(void)
{
   static struct agx_device dev;
   bool reached;

   CHECK(!open_as('G', &dev, &reached));
   CHECK(!reached && other_ioctls == 0);
   CHECK(dev.chip == AGX_CHIP_G17G);
   CHECK(!agx_is_apple9(&dev));
   CHECK(!strcmp(dev.name, "Apple M5 (G17G A0)"));

   open_as('P', &dev, &reached);
   CHECK(reached && other_ioctls == 0);
   CHECK(dev.chip == AGX_CHIP_G17P && agx_is_apple9(&dev));

   puts("PASS: G17G recognized and refused before VM setup; G17P unchanged");
   return 0;
}
