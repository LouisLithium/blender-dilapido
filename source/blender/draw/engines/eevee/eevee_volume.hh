/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup draw_engine
 *
 * Volumetric effects rendering using Frostbite's Physically-based & Unified Volumetric Rendering
 * approach.
 * https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite
 *
 * The rendering is separated in 4 stages:
 *
 * - Material Parameters : we collect volume properties of
 *   all participating media in the scene and store them in
 *   a 3D texture aligned with the 3D frustum.
 *   This is done in 2 passes, one that clear the texture
 *   and/or evaluate the world volumes, and the 2nd one that
 *   additively render object volumes.
 *
 * - Light Scattering : the volume properties then are sampled
 *   and light scattering is evaluated for each froxel of the
 *   volume texture. Temporal super-sampling (if enabled) occurs here.
 *
 * - Volume Integration : the scattered light and extinction is
 *   integrated (accumulated) along the view-rays. The result is stored
 *   for every froxel in another texture.
 *
 * - Full-screen Resolve : From the previous stage, we get two
 *   3D textures that contains integrated scattered light and extinction
 *   for "every" positions in the frustum. We only need to sample
 *   them and blend the scene color with those factors. This also
 *   work for alpha blended materials.
 */

#pragma once

#include "BLI_set.hh"

#include "DRW_gpu_wrapper.hh"
#include "GPU_batch_utils.hh"

#include "eevee_sync.hh"
#include "eevee_uniform_shared.hh"
#include "eevee_volume_shared.hh"

namespace blender::eevee {

class Instance;
class VolumePipeline;
class WorldVolumePipeline;

class VolumeModule {
  friend VolumePipeline;
  friend WorldVolumePipeline;

 private:
  Instance &inst_;

  bool enabled_;
  bool viewport_sampling_is_reset_ = false;
  bool use_reprojection_;
  bool use_lights_;

  /* Track added/removed volume objects to reset the accumulation history. */
  Set<ObjectKey> previous_objects_;
  Set<ObjectKey> current_objects_;

  VolumesInfoData &data_;

  /**
   * Occupancy map that allows to fill froxels that are inside the geometry.
   * It is filled during a pre-pass using atomic operations.
   * Using a 3D bit-field, we only allocate one bit per froxel.
   */
  Texture occupancy_tx_ = {"occupancy_tx"};
  /**
   * List of surface hit for correct occupancy determination.
   * One texture holds the number of hit count and the other the depth and
   * the facing of each hit.
   */
  Texture hit_count_tx_ = {"hit_count_tx"};
  Texture hit_depth_tx_ = {"hit_depth_tx"};
  Texture front_depth_tx_ = {"front_depth_tx"};
  Framebuffer occupancy_fb_ = {"occupancy_fb"};

  /* Material Parameters */
  Texture prop_scattering_tx_;
  Texture prop_extinction_tx_;
  Texture prop_emission_tx_;
  Texture prop_phase_tx_;
  Texture prop_phase_weight_tx_;

  /* Light Scattering. */
  PassSimple scatter_ps_ = {"Volumes.Scatter"};
  SwapChain<Texture, 2> scatter_tx_;
  SwapChain<Texture, 2> extinction_tx_;

  /* Volume Integration */
  PassSimple integration_ps_ = {"Volumes.Integration"};
  Texture integrated_scatter_tx_;
  Texture integrated_transmit_tx_;

  /* Full-screen Resolve */
  PassSimple resolve_ps_ = {"Volumes.Resolve"};
  Framebuffer resolve_fb_;

  Texture dummy_scatter_tx_;
  Texture dummy_transmit_tx_;

  View volume_view = {"Volume View"};

  float4x4 history_viewmat_ = float4x4::zero();
  /* Number of re-projected frame into the volume history.
   * Allows continuous integration between interactive and static mode. */
  int history_frame_count_ = 0;
  /* Used to detect change in camera projection type. */
  bool history_camera_is_perspective_ = false;
  /* Must be set to false on every event that makes the history invalid to sample. */
  bool valid_history_ = false;

  gpu::Batch *cube_batch_ = GPU_batch_unit_cube();

  /* ---- World space volume grid ---- */
  bool grid_wanted_ = false;
  bool grid_enabled_ = false;
  bool grid_built_ = false;
  bool grid_light_done_ = false;
  bool grid_stats_ = false;
  int3 grid_size_ = int3(0);
  Bounds<float3> grid_bounds_ = {float3(0.0f), float3(1.0f)};
  VolumesInfoData grid_volumes_data_ = {};
  View grid_view = {"Volume Grid View"};

  /* Declared before the textures so that the framebuffer outlives its attachments (C++ destroys
   * members in reverse declaration order and `Texture::~Texture` touches its attached FB). */
  Framebuffer grid_fb_ = {"Volume.GridFB"};

  Texture grid_prop_scattering_tx_ = {"grid_prop_scattering_tx"};
  Texture grid_prop_extinction_tx_ = {"grid_prop_extinction_tx"};
  Texture grid_prop_emission_tx_ = {"grid_prop_emission_tx"};
  Texture grid_prop_phase_tx_ = {"grid_prop_phase_tx"};
  Texture grid_prop_phase_weight_tx_ = {"grid_prop_phase_weight_tx"};
  /* Dummy 1x1x1 texture to satisfy the occupancy binding of the volume material shader. */
  Texture grid_dummy_occupancy_tx_ = {"grid_dummy_occupancy_tx"};
  Texture grid_scatter_tx_ = {"grid_scatter_tx"};
  Texture grid_extinct_tx_ = {"grid_extinct_tx"};
  Texture grid_depth_tx_ = {"grid_depth_tx"};

  PassSimple grid_light_ps_ = {"Volumes.GridLight"};

  /* Allocate / update the grid resources for the current frame. */
  void grid_ensure();
  /* Free the grid resources. Called at the end of the frame to avoid keeping GPU objects alive
   * across frames (they are rebuilt every frame). */
  void grid_free_resources();
  /* Read back the lit grid and print statistics (debug helper). */
  void grid_print_stats();

 public:
  VolumeModule(Instance &inst, VolumesInfoData &data) : inst_(inst), data_(data)
  {
    dummy_scatter_tx_.ensure_3d(
        gpu::TextureFormat::UNORM_8_8_8_8, int3(1), GPU_TEXTURE_USAGE_SHADER_READ, float4(0.0f));
    dummy_transmit_tx_.ensure_3d(
        gpu::TextureFormat::UNORM_8_8_8_8, int3(1), GPU_TEXTURE_USAGE_SHADER_READ, float4(1.0f));
    /* Consumers of the grid always bind something valid. */
    grid_result.scattering_tx_ = dummy_scatter_tx_;
    grid_result.extinction_tx_ = dummy_transmit_tx_;
  };

  ~VolumeModule()
  {
    GPU_BATCH_DISCARD_SAFE(cube_batch_);
  }

  /* -------------------------------------------------------------------- */
  /** \name World space volume grid (volumetric GI / shadows).
   * \{ */

  /* True if the grid is valid this frame. */
  bool grid_enabled() const
  {
    return grid_enabled_;
  }
  /* True if the world space volume grid should be built this frame (scene settings). */
  bool grid_wanted() const
  {
    return grid_wanted_;
  }
  /* World space bounds of the grid. */
  const Bounds<float3> &grid_bounds() const
  {
    return grid_bounds_;
  }

  /* Render all volume objects into the world space grid (material properties only). */
  void draw_grid_prepass();
  /* Compute the in-scattered radiance of the grid (needs shadow maps to be rendered). */
  void draw_grid_light(View &view);

  /* Uniform buffer for the grid passes. It is a copy of the main uniform data with the volume
   * view replaced by the grid view, so that materials behave identically in both passes. */
  draw::UniformBuffer<UniformData> grid_uniform_data_ = {"VolumeGridUniformDataBuf"};

  /* Grid material properties, written by the volume material passes. */
  struct {
    gpu::Texture *scattering_tx_ = nullptr;
    gpu::Texture *extinction_tx_ = nullptr;
    gpu::Texture *emission_tx_ = nullptr;
    gpu::Texture *phase_tx_ = nullptr;
    gpu::Texture *phase_weight_tx_ = nullptr;
    gpu::Texture *occupancy_tx_ = nullptr;

    template<typename PassType> void bind_resources(PassType &pass)
    {
      pass.bind_image(VOLUME_PROP_SCATTERING_IMG_SLOT, &scattering_tx_);
      pass.bind_image(VOLUME_PROP_EXTINCTION_IMG_SLOT, &extinction_tx_);
      pass.bind_image(VOLUME_PROP_EMISSION_IMG_SLOT, &emission_tx_);
      pass.bind_image(VOLUME_PROP_PHASE_IMG_SLOT, &phase_tx_);
      pass.bind_image(VOLUME_PROP_PHASE_WEIGHT_IMG_SLOT, &phase_weight_tx_);
      pass.bind_image(VOLUME_OCCUPANCY_SLOT, &occupancy_tx_);
    }
  } grid_properties;

  /* Lit grid: in-scattered radiance (RGB) and extinction (RGB). */
  struct {
    gpu::Texture *scattering_tx_ = nullptr;
    gpu::Texture *extinction_tx_ = nullptr;

    template<typename PassType> void bind_resources(PassType &pass)
    {
      pass.bind_texture(VOLUME_SCATTERING_TEX_SLOT, &scattering_tx_);
      pass.bind_texture(VOLUME_TRANSMITTANCE_TEX_SLOT, &extinction_tx_);
    }
  } grid_result;

  /** \} */

  bool needs_shadow_tagging() const
  {
    return enabled_ && use_lights_;
  }

  /* Return the future value of enabled() that will only be available after end_sync(). */
  bool will_enable() const;

  /* Returns the state of the module. */
  bool enabled() const
  {
    return enabled_;
  }

  int3 grid_size()
  {
    return data_.tex_size;
  }

  gpu::Batch *unit_cube_batch_get()
  {
    return cube_batch_;
  }

  void init();

  void begin_sync();

  void world_sync(const WorldHandle &world_handle);

  void object_sync(const ObjectHandle &ob_handle);

  void end_sync();

  void set_view(View &main_view);

  /* Render material properties. Needs to be called after `set_view`. */
  void draw_prepass(View &main_view);
  /* Compute scattering and integration. */
  void draw_compute(View &main_view, int2 extent);
  /* Final image compositing. */
  void draw_resolve(View &view);

  /* Final occupancy after resolve. Used by object volume material evaluation. */
  struct {
    /** References to the textures in the module. */
    gpu::Texture *scattering_tx_ = nullptr;
    gpu::Texture *transmittance_tx_ = nullptr;

    template<typename PassType> void bind_resources(PassType &pass)
    {
      pass.bind_texture(VOLUME_SCATTERING_TEX_SLOT, &scattering_tx_);
      pass.bind_texture(VOLUME_TRANSMITTANCE_TEX_SLOT, &transmittance_tx_);
    }
  } result;

  /* Volume property buffers that are populated by objects or world volume shaders. */
  struct {
    /** References to the textures in the module. */
    gpu::Texture *scattering_tx_ = nullptr;
    gpu::Texture *extinction_tx_ = nullptr;
    gpu::Texture *emission_tx_ = nullptr;
    gpu::Texture *phase_tx_ = nullptr;
    gpu::Texture *phase_weight_tx_ = nullptr;
    gpu::Texture *occupancy_tx_ = nullptr;

    template<typename PassType> void bind_resources(PassType &pass)
    {
      pass.bind_image(VOLUME_PROP_SCATTERING_IMG_SLOT, &scattering_tx_);
      pass.bind_image(VOLUME_PROP_EXTINCTION_IMG_SLOT, &extinction_tx_);
      pass.bind_image(VOLUME_PROP_EMISSION_IMG_SLOT, &emission_tx_);
      pass.bind_image(VOLUME_PROP_PHASE_IMG_SLOT, &phase_tx_);
      pass.bind_image(VOLUME_PROP_PHASE_WEIGHT_IMG_SLOT, &phase_weight_tx_);
      pass.bind_image(VOLUME_OCCUPANCY_SLOT, &occupancy_tx_);
    }
  } properties;

  /* Textures used for object volume occupancy computation. */
  struct {
    /** References to the textures in the module. */
    gpu::Texture *occupancy_tx_ = nullptr;
    gpu::Texture *hit_depth_tx_ = nullptr;
    gpu::Texture *hit_count_tx_ = nullptr;

    template<typename PassType> void bind_resources(PassType &pass)
    {
      pass.bind_image(VOLUME_OCCUPANCY_SLOT, &occupancy_tx_);
      pass.bind_image(VOLUME_HIT_DEPTH_SLOT, &hit_depth_tx_);
      pass.bind_image(VOLUME_HIT_COUNT_SLOT, &hit_count_tx_);
    }
  } occupancy;
};
}  // namespace blender::eevee
