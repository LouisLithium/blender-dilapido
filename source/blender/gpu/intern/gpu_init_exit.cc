/* SPDX-FileCopyrightText: 2013 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "BKE_material.hh"

#include "BLI_vector.hh"

#include "GPU_batch.hh"
#include "GPU_init_exit.hh" /* interface */
#include "GPU_pass.hh"

#include "intern/gpu_private.hh"
#include "intern/gpu_shader_create_info_private.hh"
#include "intern/gpu_shader_dependency_private.hh"

namespace blender {

/**
 * although the order of initialization and shutdown should not matter
 * (except for the extensions), I chose alphabetical and reverse alphabetical order
 */
static bool initialized = false;

namespace {

struct ExitHook {
  void (*callback)(void *user_data);
  void *user_data;
};

Vector<ExitHook> &exit_hooks()
{
  static Vector<ExitHook> hooks;
  return hooks;
}

}  // namespace

void GPU_exit_hook_register(void (*callback)(void *user_data), void *user_data)
{
  exit_hooks().append({callback, user_data});
}

void GPU_exit_hook_unregister(void (*callback)(void *user_data), const void *user_data)
{
  for (const int i : exit_hooks().index_range()) {
    if (exit_hooks()[i].callback == callback && exit_hooks()[i].user_data == user_data) {
      exit_hooks().remove(i);
      return;
    }
  }
}

void GPU_init()
{
  /* can't avoid calling this multiple times, see wm_window_ghostwindow_add */
  if (initialized) {
    return;
  }

  initialized = true;

  gpu_backend_init_resources();

  gpu_shader_dependency_init();
  gpu_shader_create_info_init();

  GPU_shader_builtin_warm_up();
  GPU_pass_cache_init();

  gpu_batch_init();
}

void GPU_exit()
{
  /* Modules with their own context have to let go of it first. */
  {
    const Vector<ExitHook> hooks = exit_hooks();
    exit_hooks().clear();
    for (const ExitHook &hook : hooks) {
      hook.callback(hook.user_data);
    }
  }

  gpu_batch_exit();

  GPU_pass_cache_free();

  BKE_material_defaults_free_gpu();
  GPU_shader_free_builtin_shaders();

  gpu_backend_delete_resources();

  gpu_shader_dependency_exit();
  gpu_shader_create_info_exit();

  initialized = false;
}

bool GPU_is_init()
{
  return initialized;
}

}  // namespace blender
