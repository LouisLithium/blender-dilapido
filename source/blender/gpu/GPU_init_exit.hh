/* SPDX-FileCopyrightText: 2013 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#pragma once

namespace blender {

void GPU_init();
void GPU_exit();
bool GPU_is_init();

/**
 * Runs \a callback from #GPU_exit before any device resources are released.
 *
 * Modules that keep a GPU context of their own (the `HW Accelerate` backend does) have to dispose
 * of it while the backend is still alive: a context that outlives #GPU_exit leaves its allocations
 * behind, which the Vulkan allocator reports as unfreed and then crashes on.
 */
void GPU_exit_hook_register(void (*callback)(void *user_data), void *user_data);
void GPU_exit_hook_unregister(void (*callback)(void *user_data), const void *user_data);

}  // namespace blender
