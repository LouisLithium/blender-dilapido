/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup draw_engine
 *
 * Volumetric effects rendering using Frostbite's Physically-based & Unified Volumetric Rendering
 * approach.
 * https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite
 */

#include "GPU_capabilities.hh"

#include "GPU_debug.hh"
#include "GPU_state.hh"
#include "GPU_texture.hh"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "eevee_instance.hh"
#include "eevee_pipeline.hh"

#include "eevee_volume.hh"

namespace blender::eevee {

void VolumeModule::init()
{
  enabled_ = false;

  const Scene *scene_eval = inst_.scene;

  const int2 extent = inst_.film.render_extent_get();
  int tile_size = clamp_i(scene_eval->eevee.volumetric_tile_size, 1, 16);

  int3 tex_size;
  /* Try to match resolution setting but fall back to lower resolution
   * if it doesn't fit the hardware limits. */
  for (; tile_size <= 16; tile_size *= 2) {
    /* Find Froxel Texture resolution. */
    tex_size = int3(math::divide_ceil(extent, int2(tile_size)), 0);
    tex_size.z = std::max(1, scene_eval->eevee.volumetric_samples);

    if (math::reduce_max(tex_size) < GPU_max_texture_3d_size()) {
      /* Fits hardware limits. */
      break;
    }
  }

  if (tile_size != scene_eval->eevee.volumetric_tile_size) {
    inst_.info_append_i18n(
        "Warning: Volume rendering data could not be allocated. Now using a resolution of 1:{} "
        "instead of 1:{}.",
        tile_size,
        scene_eval->eevee.volumetric_tile_size);
  }

  data_.tile_size = tile_size;
  data_.tile_size_lod = int(log2(tile_size));
  data_.coord_scale = float2(extent) / float2(tile_size * tex_size);
  data_.main_view_extent = float2(extent);
  data_.main_view_extent_inv = 1.0f / float2(extent);
  data_.tex_size = tex_size;
  data_.inv_tex_size = 1.0f / float3(tex_size);

  const bool shadow_enabled = (scene_eval->eevee.flag & SCE_EEVEE_VOLUMETRIC_SHADOWS) != 0;
  data_.shadow_steps = (shadow_enabled) ? scene_eval->eevee.volumetric_shadow_samples : 0;

  data_.light_clamp = scene_eval->eevee.volumetric_light_clamp;
  data_.light_clamp = (data_.light_clamp > 0.0) ? data_.light_clamp : 1e20;

  use_reprojection_ = (scene_eval->eevee.flag & SCE_EEVEE_TAA_REPROJECTION) != 0;

  /* Debug: print statistics about the world space volume grid (host readback). */
  grid_stats_ = getenv("BLENDER_EEVEE_VOLUME_GRID_STATS") != nullptr;
}

void VolumeModule::begin_sync()
{
  previous_objects_ = current_objects_;
  current_objects_.clear();

  /* Sync runs before we know the volume bounds, but the scene settings are enough to know if the
   * world space volume grid will be built this frame. */
  const SceneEEVEE &eevee = inst_.scene->eevee;
  grid_wanted_ = (eevee.flag & (SCE_EEVEE_VOLUMETRIC_GI | SCE_EEVEE_VOLUMETRIC_SHADOWS_SURFACE)) != 0 ||
                 (getenv("BLENDER_EEVEE_VOLUME_GRID") != nullptr);
}

void VolumeModule::world_sync(const WorldHandle &world_handle)
{
  if (!use_reprojection_) {
    return;
  }

  if (world_handle.recalc && !inst_.is_playback) {
    valid_history_ = false;
  }
}

void VolumeModule::object_sync(const ObjectHandle &ob_handle)
{
  for (int i : IndexRange(ob_handle.instances_count())) {
    current_objects_.add(ObjectKey(ob_handle, i));
  }

  if (!use_reprojection_) {
    return;
  }

  if (ob_handle.recalc && !inst_.is_playback) {
    valid_history_ = false;
  }
}

bool VolumeModule::will_enable() const
{
  return inst_.world.has_volume() || !current_objects_.is_empty() ||
         inst_.film.get_data().volume_light_id != -1;
}

void VolumeModule::end_sync()
{
  enabled_ = will_enable();
  /* Save reset state. */
  viewport_sampling_is_reset_ = inst_.is_viewport() && inst_.sampling.is_reset();

  const Scene *scene_eval = inst_.scene;

  const bool custom_range = scene_eval->eevee.flag & SCE_EEVEE_VOLUME_CUSTOM_RANGE;
  const float camera_clip_start = inst_.camera.data_get().clip_near;
  const float camera_clip_end = inst_.camera.data_get().clip_far;
  float integration_start = custom_range ? scene_eval->eevee.volumetric_start : camera_clip_start;
  float integration_end = custom_range ? scene_eval->eevee.volumetric_end : camera_clip_end;

  if (!inst_.camera.is_camera_object() && inst_.camera.is_orthographic()) {
    integration_start = -integration_end;
  }

  std::optional<Bounds<float>> volume_bounds = inst_.pipelines.volume.object_integration_range();
  if (volume_bounds && !inst_.world.has_volume()) {
    /* Restrict integration range to the object volume range. This increases precision. */
    integration_start = math::max(integration_start, -volume_bounds.value().max);
    integration_end = math::min(integration_end, -volume_bounds.value().min);
  }

  /* Negate clip values (View matrix forward vector is -Z). */
  float near = -math::max(integration_start, camera_clip_start - 1e-4f);
  float far = -math::min(integration_end, camera_clip_end + 1e-4f);

  if (assign_if_different(history_camera_is_perspective_, inst_.camera.is_perspective())) {
    /* Currently, the re-projection uses the same path for volume_z_to_view_z conversion for both
     * the current view and the history view. Moreover, re-projecting in this huge change is more
     * detrimental than anything. */
    valid_history_ = false;
  }

  if (valid_history_) {
    /* Avoid the (potentially expensive) check if valid_history_ is already false. */
    if (current_objects_ != previous_objects_) {
      valid_history_ = false;
    }
  }

  if (inst_.camera.is_perspective()) {
    float sample_distribution = scene_eval->eevee.volumetric_sample_distribution;
    sample_distribution = 4.0f * math::max(1.0f - sample_distribution, 1e-2f);

    data_.depth_near = (far - near * exp2(1.0f / sample_distribution)) / (far - near);
    data_.depth_far = (1.0f - data_.depth_near) / near;
    data_.depth_distribution = sample_distribution;
  }
  else {
    data_.depth_near = near;
    data_.depth_far = far;
    data_.depth_distribution = 0.0f; /* Unused. */
  }

  if (!enabled_) {
    occupancy_tx_.free();
    prop_scattering_tx_.free();
    prop_extinction_tx_.free();
    prop_emission_tx_.free();
    prop_phase_tx_.free();
    prop_phase_weight_tx_.free();
    scatter_tx_.current().free();
    scatter_tx_.previous().free();
    extinction_tx_.current().free();
    extinction_tx_.previous().free();
    integrated_scatter_tx_.free();
    integrated_transmit_tx_.free();

    /* World space grid. */
    grid_enabled_ = false;
    grid_built_ = false;
    grid_light_done_ = false;
    grid_prop_scattering_tx_.free();
    grid_prop_extinction_tx_.free();
    grid_prop_emission_tx_.free();
    grid_prop_phase_tx_.free();
    grid_prop_phase_weight_tx_.free();
    grid_dummy_occupancy_tx_.free();
    grid_scatter_tx_.free();
    grid_extinct_tx_.free();
    grid_depth_tx_.free();

    /* Make sure the consumers do not sample a stale grid. */
    grid_result.scattering_tx_ = dummy_scatter_tx_;
    grid_result.extinction_tx_ = dummy_transmit_tx_;
    inst_.uniform_data.data.volume_grid.enabled = 0.0f;
    inst_.uniform_data.data.volume_grid.bounds_size = float3(0.0f);

    /* Update references for bindings. */
    result.scattering_tx_ = dummy_scatter_tx_;
    result.transmittance_tx_ = dummy_transmit_tx_;
    /* These shouldn't be used. */
    properties.scattering_tx_ = nullptr;
    properties.extinction_tx_ = nullptr;
    properties.emission_tx_ = nullptr;
    properties.phase_tx_ = nullptr;
    properties.phase_weight_tx_ = nullptr;
    properties.occupancy_tx_ = nullptr;
    occupancy.occupancy_tx_ = nullptr;
    occupancy.hit_depth_tx_ = nullptr;
    occupancy.hit_count_tx_ = nullptr;

    /* Avoid undefined re-projection behavior. */
    valid_history_ = false;
    return;
  }

  bool has_scatter = inst_.world.has_volume_scatter() || inst_.pipelines.volume.has_scatter();
  bool has_absorption = inst_.world.has_volume_absorption() ||
                        inst_.pipelines.volume.has_absorption();
  use_lights_ = has_scatter;
  /* TODO(fclem): Allocate extinction texture as dummy (1px^3) if has_absorption are false. */
  UNUSED_VARS(has_absorption);

  eGPUTextureUsage usage = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE |
                           GPU_TEXTURE_USAGE_ATTACHMENT;

  prop_scattering_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  prop_extinction_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  prop_emission_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  /* We need 2 separate images to prevent bugs in Nvidia drivers (See #122454). */
  prop_phase_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16, data_.tex_size, usage);
  prop_phase_weight_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16, data_.tex_size, usage);

  int occupancy_layers = divide_ceil_u(data_.tex_size.z, 32u);
  eGPUTextureUsage occupancy_usage = GPU_TEXTURE_USAGE_SHADER_READ |
                                     GPU_TEXTURE_USAGE_SHADER_WRITE | GPU_TEXTURE_USAGE_ATOMIC;
  occupancy_tx_.ensure_3d(
      gpu::TextureFormat::UINT_32, int3(data_.tex_size.xy(), occupancy_layers), occupancy_usage);

  {
    eGPUTextureUsage hit_count_usage = GPU_TEXTURE_USAGE_SHADER_READ |
                                       GPU_TEXTURE_USAGE_SHADER_WRITE | GPU_TEXTURE_USAGE_ATOMIC;
    eGPUTextureUsage hit_depth_usage = GPU_TEXTURE_USAGE_SHADER_READ |
                                       GPU_TEXTURE_USAGE_SHADER_WRITE;
    int2 hit_list_size = int2(1);
    int hit_list_layer = 1;
    if (inst_.pipelines.volume.use_hit_list()) {
      hit_list_layer = clamp_i(inst_.scene->eevee.volumetric_ray_depth, 1, 16);
      hit_list_size = data_.tex_size.xy();
    }
    hit_depth_tx_.ensure_3d(
        gpu::TextureFormat::SFLOAT_32, int3(hit_list_size, hit_list_layer), hit_depth_usage);
    if (hit_count_tx_.ensure_2d(gpu::TextureFormat::UINT_32, hit_list_size, hit_count_usage)) {
      hit_count_tx_.clear(uint4(0u));
    }
  }

  eGPUTextureUsage front_depth_usage = GPU_TEXTURE_USAGE_SHADER_READ |
                                       GPU_TEXTURE_USAGE_ATTACHMENT;
  front_depth_tx_.ensure_2d(
      gpu::TextureFormat::SFLOAT_32_DEPTH_UINT_8, data_.tex_size.xy(), front_depth_usage);
  occupancy_fb_.ensure(GPU_ATTACHMENT_TEXTURE(front_depth_tx_));

  bool created = false;
  created |= scatter_tx_.current().ensure_3d(
      gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  created |= extinction_tx_.current().ensure_3d(
      gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  created |= scatter_tx_.previous().ensure_3d(
      gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  created |= extinction_tx_.previous().ensure_3d(
      gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);

  if (created) {
    valid_history_ = false;
  }

  integrated_scatter_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);
  integrated_transmit_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, data_.tex_size, usage);

  /* Update references for bindings. */
  result.scattering_tx_ = integrated_scatter_tx_;
  result.transmittance_tx_ = integrated_transmit_tx_;
  properties.scattering_tx_ = prop_scattering_tx_;
  properties.extinction_tx_ = prop_extinction_tx_;
  properties.emission_tx_ = prop_emission_tx_;
  properties.phase_tx_ = prop_phase_tx_;
  properties.phase_weight_tx_ = prop_phase_weight_tx_;
  properties.occupancy_tx_ = occupancy_tx_;
  occupancy.occupancy_tx_ = occupancy_tx_;
  occupancy.hit_depth_tx_ = hit_depth_tx_;
  occupancy.hit_count_tx_ = hit_count_tx_;

  /* Set extend mode to extend and reject invalid samples in the shader.
   * This avoids some black rim artifacts near the edge of the re-projected volume.
   * Filter linear to avoid sharp artifacts during re-projection. */
  const GPUSamplerState history_sampler = {GPU_SAMPLER_FILTERING_LINEAR,
                                           GPU_SAMPLER_EXTEND_MODE_EXTEND,
                                           GPU_SAMPLER_EXTEND_MODE_EXTEND};
  scatter_ps_.init();
  scatter_ps_.shader_set(
      inst_.shaders.static_shader_get(use_lights_ ? VOLUME_SCATTER_WITH_LIGHTS : VOLUME_SCATTER));
  scatter_ps_.bind_resources(inst_.hiz_buffer.front);
  scatter_ps_.bind_resources(inst_.lights);
  scatter_ps_.bind_resources(inst_.sphere_probes);
  scatter_ps_.bind_resources(inst_.volume_probes);
  scatter_ps_.bind_resources(inst_.shadows);
  scatter_ps_.bind_resources(inst_.uniform_data);
  scatter_ps_.bind_resources(inst_.sampling);
  scatter_ps_.bind_image("in_scattering_img", &prop_scattering_tx_);
  scatter_ps_.bind_image("in_extinction_img", &prop_extinction_tx_);
  scatter_ps_.bind_texture("extinction_tx", &prop_extinction_tx_);
  scatter_ps_.bind_image("in_emission_img", &prop_emission_tx_);
  scatter_ps_.bind_image("in_phase_img", &prop_phase_tx_);
  scatter_ps_.bind_image("in_phase_weight_img", &prop_phase_weight_tx_);
  scatter_ps_.bind_texture("scattering_history_tx", &scatter_tx_.previous(), history_sampler);
  scatter_ps_.bind_texture("extinction_history_tx", &extinction_tx_.previous(), history_sampler);
  scatter_ps_.bind_image("out_scattering_img", &scatter_tx_.current());
  scatter_ps_.bind_image("out_extinction_img", &extinction_tx_.current());
  scatter_ps_.bind_texture(RBUFS_UTILITY_TEX_SLOT, inst_.pipelines.utility_tx);
  /* Sync with the property pass. */
  scatter_ps_.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH);
  scatter_ps_.dispatch(math::divide_ceil(data_.tex_size, int3(VOLUME_GROUP_SIZE)));

  integration_ps_.init();
  integration_ps_.shader_set(inst_.shaders.static_shader_get(VOLUME_INTEGRATION));
  integration_ps_.bind_resources(inst_.uniform_data);
  integration_ps_.bind_resources(inst_.sampling);
  integration_ps_.bind_texture("in_scattering_tx", &scatter_tx_.current());
  integration_ps_.bind_texture("in_extinction_tx", &extinction_tx_.current());
  integration_ps_.bind_image("out_scattering_img", &integrated_scatter_tx_);
  integration_ps_.bind_image("out_transmittance_img", &integrated_transmit_tx_);
  /* Sync with the scatter pass. */
  integration_ps_.barrier(GPU_BARRIER_TEXTURE_FETCH);
  integration_ps_.dispatch(
      math::divide_ceil(int2(data_.tex_size), int2(VOLUME_INTEGRATION_GROUP_SIZE)));

  resolve_ps_.init();
  resolve_ps_.state_set(DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_CUSTOM);
  resolve_ps_.shader_set(inst_.shaders.static_shader_get(VOLUME_RESOLVE));
  resolve_ps_.bind_resources(inst_.uniform_data);
  resolve_ps_.bind_resources(this->result);
  resolve_ps_.bind_resources(inst_.hiz_buffer.front);
  resolve_ps_.bind_image(RBUFS_COLOR_SLOT, &inst_.render_buffers.rp_color_tx);
  resolve_ps_.bind_image(RBUFS_VALUE_SLOT, &inst_.render_buffers.rp_value_tx);
  /* Sync with the integration pass. */
  resolve_ps_.barrier(GPU_BARRIER_TEXTURE_FETCH);
  resolve_ps_.draw_procedural(GPU_PRIM_TRIS, 1, 3);

  grid_ensure();
}

void VolumeModule::set_view(View &main_view)
{
  /* Number of frame to consider for blending with exponential (infinite) average. */
  int exponential_frame_count = 16;
  if (inst_.is_image_render) {
    /* Disable reprojection for rendering. */
    exponential_frame_count = 0;
  }
  else if (!use_reprojection_) {
    /* No re-projection if TAA is disabled. */
    exponential_frame_count = 0;
  }
  else if (inst_.is_playback) {
    /* For now, we assume we want responsiveness for volume animation.
     * But this makes general animation inside uniform volumes less stable.
     * When we introduce updated volume tagging, we will be able to increase this for general
     * playback. */
    exponential_frame_count = 3;
  }
  else if (inst_.is_transforming) {
    /* Improve responsiveness of volume if we are transforming objects. */
    /* TODO(fclem): This is too general as it will be triggered even for non volume object.
     * Instead, we should tag which areas of the volume that need increased responsiveness. */
    exponential_frame_count = 3;
  }
  else if (inst_.is_navigating) {
    /* Navigation is usually smooth because of the re-projection but we can get ghosting
     * artifacts on lights because of voxels stretched in Z or anisotropy. */
    exponential_frame_count = 8;
  }
  else if (viewport_sampling_is_reset_) {
    /* If we are not falling in any cases above, this usually means there is a scene or object
     * parameter update. Reset accumulation completely. */
    exponential_frame_count = 0;
  }

  if (!valid_history_) {
    history_frame_count_ = 0;
  }
  /* Interactive mode accumulate samples using exponential average.
   * We still reuse the history when we go into static mode.
   * However, using re-projection for static mode will show the precision limit of RG11B10 format.
   * So we clamp it to the exponential frame count in any case. */
  history_frame_count_ = math::min(history_frame_count_, exponential_frame_count);

  /* In interactive mode, use exponential average (fixed ratio).
   * For static / render mode use simple average (moving ratio). */
  float history_opacity = history_frame_count_ / (history_frame_count_ + 1.0f);

  /* Setting opacity to 0.0 will bypass any sampling of history buffer.
   * Allowing us to skip the 3D texture clear. */
  data_.history_opacity = (valid_history_) ? history_opacity : 0.0f;

  float left, right, bottom, top, near, far;
  projmat_dimensions(main_view.winmat().ptr(), &left, &right, &bottom, &top, &near, &far);

  /* Just like up-sampling matrix computation, we have to be careful to where to put the bounds of
   * our froxel volume so that a 2D pixel covers exactly the number of pixel in a tile. */
  float2 render_size = float2(right - left, top - bottom);
  float2 volume_size = render_size * float2(data_.tex_size.xy() * data_.tile_size) /
                       float2(inst_.film.render_extent_get());
  /* Change to the padded extends. */
  right = left + volume_size.x;
  top = bottom + volume_size.y;

  float4x4 winmat_infinite, winmat_finite;
  /* Create an infinite projection matrix to avoid far clipping plane clipping the object. This
   * way, surfaces that are further away than the far clip plane will still be voxelized. */
  winmat_infinite = main_view.is_persp() ?
                        math::projection::perspective_infinite(left, right, bottom, top, near) :
                        math::projection::orthographic_infinite(left, right, bottom, top, near);
  /* We still need a bounded projection matrix to get correct froxel location. */
  winmat_finite = main_view.is_persp() ?
                      math::projection::perspective(left, right, bottom, top, near, far) :
                      math::projection::orthographic(left, right, bottom, top, near, far);
  /* Save non-jittered finite matrix for re-projection. */
  data_.winmat_stable = winmat_finite;
  data_.wininv_stable = math::invert(winmat_finite);

  /* Anti-Aliasing / Super-Sampling jitter. */
  float2 jitter = inst_.sampling.rng_2d_get(SAMPLING_VOLUME_U);
  /* Wrap to keep first sample centered (0,0) and scale to convert to NDC. */
  jitter = math::fract(jitter + 0.5f) * 2.0f - 1.0f;
  /* Convert to pixel size. */
  jitter *= data_.inv_tex_size.xy();
  /* Apply jitter to both matrices. */
  winmat_infinite = math::projection::translate(winmat_infinite, jitter);
  winmat_finite = math::projection::translate(winmat_finite, jitter);

  data_.winmat_finite = winmat_finite;
  data_.wininv_finite = math::invert(winmat_finite);

  /* Compute re-projection matrix. */
  data_.curr_view_to_past_view = history_viewmat_ * main_view.viewinv();

  volume_view.sync(main_view.viewmat(), winmat_infinite);
}

void VolumeModule::grid_ensure()
{
  grid_enabled_ = false;
  grid_built_ = false;
  grid_light_done_ = false;
  inst_.uniform_data.data.volume_grid.enabled = 0.0f;

  std::optional<Bounds<float3>> bounds = inst_.pipelines.volume.object_world_bounds();
  /* The world space volume grid feeds the volumetric GI and the volumetric shadows on surfaces. */
  const SceneEEVEE &eevee = inst_.scene->eevee;
  const bool grid_requested = grid_wanted_;
  if (!enabled_ || !grid_requested || !bounds.has_value()) {
    grid_free_resources();
    return;
  }

  Bounds<float3> b = *bounds;
  float3 extent = math::max(b.max - b.min, float3(1e-3f));
  /* Small padding so that the grid does not clip the volume at its borders. */
  const float3 pad = math::max(extent * 0.01f, float3(1e-4f));
  b.min -= pad;
  b.max += pad;
  extent = b.max - b.min;

  int res = eevee.volumetric_grid_resolution;
  if (const char *env = getenv("BLENDER_EEVEE_VOLUME_GRID_RES")) {
    res = atoi(env);
  }
  /* Old files have a zeroed value for the newly added field. */
  if (res == 0) {
    res = 96;
  }
  res = clamp_i(res, 8, 256);

  /* Cubic voxels. */
  const float voxel_size = math::reduce_max(extent) / float(res);
  int3 size = int3(int(ceilf(extent.x / voxel_size)),
                   int(ceilf(extent.y / voxel_size)),
                   int(ceilf(extent.z / voxel_size)));
  size = math::max(size, int3(1));
  size = math::min(size, int3(res));

  grid_bounds_ = b;
  grid_size_ = size;

  const float3 center = (b.min + b.max) * 0.5f;
  const float eps = voxel_size * 0.5f;
  /* Orthographic view looking down -Z: froxel Z=0 is at the grid max Z. */
  const float4x4 viewmat = math::from_location<float4x4>(
      float3(-center.x, -center.y, -(b.max.z + eps)));
  const float4x4 winmat = math::projection::orthographic(
      -extent.x * 0.5f, extent.x * 0.5f, -extent.y * 0.5f, extent.y * 0.5f, eps, extent.z + eps);
  grid_view.sync(viewmat, winmat);

  VolumesInfoData &gd = grid_volumes_data_;
  gd = {};
  gd.tex_size = size;
  gd.inv_tex_size = 1.0f / float3(size);
  gd.winmat_finite = winmat;
  gd.wininv_finite = math::invert(winmat);
  gd.winmat_stable = winmat;
  gd.wininv_stable = math::invert(winmat);
  /* View space Z (negative, as the view looks down -Z). */
  gd.depth_near = -eps;
  gd.depth_far = -(extent.z + eps);
  gd.depth_distribution = 0.0f;
  gd.history_depth_near = gd.depth_near;
  gd.history_depth_far = gd.depth_far;
  gd.history_depth_distribution = 0.0f;
  gd.history_opacity = 0.0f;
  gd.tile_size = 1;
  gd.tile_size_lod = 0;
  gd.coord_scale = float2(1.0f);
  gd.main_view_extent = float2(size.xy());
  gd.main_view_extent_inv = 1.0f / float2(size.xy());
  gd.light_clamp = data_.light_clamp;
  gd.shadow_steps = 0.0f;

  const eGPUTextureUsage usage = GPU_TEXTURE_USAGE_SHADER_READ |
                                 GPU_TEXTURE_USAGE_SHADER_WRITE;
  grid_prop_scattering_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, size, usage);
  grid_prop_extinction_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, size, usage);
  grid_prop_emission_tx_.ensure_3d(gpu::TextureFormat::UFLOAT_11_11_10, size, usage);
  grid_prop_phase_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16, size, usage);
  grid_prop_phase_weight_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16, size, usage);
  grid_dummy_occupancy_tx_.ensure_3d(gpu::TextureFormat::UINT_32,
                                     int3(1),
                                     usage | GPU_TEXTURE_USAGE_ATOMIC);
  grid_scatter_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16_16_16_16,
                             size,
                             usage | GPU_TEXTURE_USAGE_HOST_READ);
  grid_extinct_tx_.ensure_3d(gpu::TextureFormat::SFLOAT_16_16_16_16,
                             size,
                             usage | GPU_TEXTURE_USAGE_HOST_READ);
  grid_depth_tx_.ensure_2d(gpu::TextureFormat::SFLOAT_32_DEPTH_UINT_8,
                           size.xy(),
                           GPU_TEXTURE_USAGE_ATTACHMENT);
  grid_fb_.ensure(GPU_ATTACHMENT_TEXTURE(grid_depth_tx_));

  grid_properties.scattering_tx_ = grid_prop_scattering_tx_;
  grid_properties.extinction_tx_ = grid_prop_extinction_tx_;
  grid_properties.emission_tx_ = grid_prop_emission_tx_;
  grid_properties.phase_tx_ = grid_prop_phase_tx_;
  grid_properties.phase_weight_tx_ = grid_prop_phase_weight_tx_;
  grid_properties.occupancy_tx_ = grid_dummy_occupancy_tx_;
  grid_result.scattering_tx_ = grid_scatter_tx_;
  grid_result.extinction_tx_ = grid_extinct_tx_;

  inst_.uniform_data.data.volume_grid.bounds_min = b.min;
  inst_.uniform_data.data.volume_grid.bounds_size = extent;
  inst_.uniform_data.data.volume_grid.mode = 0;
  inst_.uniform_data.data.volume_grid.enabled = 1.0f;
  int gi_samples = eevee.volumetric_grid_samples;
  inst_.uniform_data.data.volume_grid.samples = float4(
      float(gi_samples > 0 ? gi_samples : 8),
      float(clamp_i(eevee.volumetric_shadow_samples, 1, 128)),
      0.0f,
      0.0f);

  grid_light_ps_.init();
  grid_light_ps_.shader_set(inst_.shaders.static_shader_get(VOLUME_GRID_LIGHT));
  grid_light_ps_.bind_resources(inst_.uniform_data);
  grid_light_ps_.bind_resources(inst_.lights);
  grid_light_ps_.bind_image(VOLUME_PROP_SCATTERING_IMG_SLOT, &grid_prop_scattering_tx_);
  grid_light_ps_.bind_image(VOLUME_PROP_EXTINCTION_IMG_SLOT, &grid_prop_extinction_tx_);
  grid_light_ps_.bind_image(VOLUME_PROP_EMISSION_IMG_SLOT, &grid_prop_emission_tx_);
  grid_light_ps_.bind_image(6, &grid_scatter_tx_);
  grid_light_ps_.bind_image(7, &grid_extinct_tx_);
  grid_light_ps_.barrier(GPU_BARRIER_SHADER_IMAGE_ACCESS);
  grid_light_ps_.dispatch(math::divide_ceil(size, int3(VOLUME_GROUP_SIZE)));
  /* The consumers (GI rays, surface lighting) sample these images as textures. */
  grid_light_ps_.barrier(GPU_BARRIER_TEXTURE_FETCH);

  grid_enabled_ = true;
}

void VolumeModule::grid_free_resources()
{
  if (!grid_enabled_ && !grid_depth_tx_.is_valid()) {
    return;
  }
  /* Clear the binding pointers first, then detach and free. */
  grid_properties.scattering_tx_ = nullptr;
  grid_properties.extinction_tx_ = nullptr;
  grid_properties.emission_tx_ = nullptr;
  grid_properties.phase_tx_ = nullptr;
  grid_properties.phase_weight_tx_ = nullptr;
  grid_properties.occupancy_tx_ = nullptr;
  grid_result.scattering_tx_ = dummy_scatter_tx_;
  grid_result.extinction_tx_ = dummy_transmit_tx_;
  /* Detach the framebuffer before freeing the attachment. */
  grid_fb_.ensure(GPU_ATTACHMENT_NONE);
  grid_prop_scattering_tx_.free();
  grid_prop_extinction_tx_.free();
  grid_prop_emission_tx_.free();
  grid_prop_phase_tx_.free();
  grid_prop_phase_weight_tx_.free();
  grid_dummy_occupancy_tx_.free();
  grid_scatter_tx_.free();
  grid_extinct_tx_.free();
  grid_depth_tx_.free();
  grid_enabled_ = false;
  /* Fall back to dummy textures so that the consumers can bind something valid. */
  grid_result.scattering_tx_ = dummy_scatter_tx_;
  grid_result.extinction_tx_ = dummy_transmit_tx_;
  inst_.uniform_data.data.volume_grid.enabled = 0.0f;
  inst_.uniform_data.data.volume_grid.bounds_size = float3(0.0f);
}

void VolumeModule::draw_grid_prepass()
{
  if (!grid_enabled_ || grid_built_) {
    return;
  }
  grid_built_ = true;

  GPU_debug_group_begin("Volume.Grid");

  grid_prop_scattering_tx_.clear(float4(0.0f));
  grid_prop_extinction_tx_.clear(float4(0.0f));
  grid_prop_emission_tx_.clear(float4(0.0f));
  grid_prop_phase_tx_.clear(float4(0.0f));
  grid_prop_phase_weight_tx_.clear(float4(0.0f));
  grid_dummy_occupancy_tx_.clear(uint4(0u));

  /* Copy the main uniform data so that materials behave identically in both passes, then
   * override the volume view with the grid view. */
  grid_uniform_data_ = static_cast<const UniformData &>(inst_.uniform_data.data);
  grid_uniform_data_.volumes = grid_volumes_data_;
  grid_uniform_data_.volume_grid.bounds_min = grid_bounds_.min;
  grid_uniform_data_.volume_grid.bounds_size = grid_bounds_.max - grid_bounds_.min;
  grid_uniform_data_.volume_grid.mode = 1;
  grid_uniform_data_.push_update();

  grid_fb_.bind();
  grid_view.visibility_test(false);
  inst_.pipelines.volume.render_grid(grid_view);

  GPU_debug_group_end();
}

void VolumeModule::draw_grid_light(View &view)
{
  if (!grid_enabled_ || grid_light_done_) {
    return;
  }
  grid_light_done_ = true;

  inst_.manager->submit(grid_light_ps_, view);

  if (grid_stats_) {
    grid_print_stats();
  }
}

void VolumeModule::grid_print_stats()
{
  /* Debug helper: read back the lit grid and print statistics. This is the reference used to
   * validate the world space lighting (M1) and the GI integration (M2). Only used by the
   * regression probes, hence the GPU stall. */
  GPU_finish();
  const int3 size = grid_size_;
  const int64_t voxel_count = int64_t(size.x) * size.y * size.z;

  const size_t scatter_bytes = GPU_texture_read_size_get(grid_scatter_tx_, GPU_DATA_FLOAT, 0);
  const size_t extinct_bytes = GPU_texture_read_size_get(grid_extinct_tx_, GPU_DATA_FLOAT, 0);
  if (scatter_bytes == 0 || extinct_bytes == 0) {
    printf("VOLUME_GRID_STATS: readback failed\n");
    fflush(stdout);
    return;
  }

  std::vector<float> scatter(scatter_bytes / sizeof(float));
  std::vector<float> extinct(extinct_bytes / sizeof(float));
  GPU_texture_read(grid_scatter_tx_, GPU_DATA_FLOAT, 0, scatter.data());
  GPU_texture_read(grid_extinct_tx_, GPU_DATA_FLOAT, 0, extinct.data());

  double scatter_sum = 0.0;
  double extinct_sum = 0.0;
  double extinct_max = 0.0;
  int occupied = 0;
  for (int64_t i = 0; i < voxel_count; i++) {
    const float sr = scatter[i * 4 + 0];
    const float sg = scatter[i * 4 + 1];
    const float sb = scatter[i * 4 + 2];
    const float et = extinct[i * 4 + 0];
    scatter_sum += double(sr + sg + sb) / 3.0;
    extinct_sum += et;
    extinct_max = math::max(extinct_max, double(et));
    occupied += (et > 1e-6f) ? 1 : 0;
  }
  printf("VOLUME_GRID_STATS: size=(%d %d %d) bounds_min=(%.4f %.4f %.4f) bounds_size=(%.4f %.4f %.4f)\n",
         size.x,
         size.y,
         size.z,
         grid_bounds_.min.x,
         grid_bounds_.min.y,
         grid_bounds_.min.z,
         grid_bounds_.max.x - grid_bounds_.min.x,
         grid_bounds_.max.y - grid_bounds_.min.y,
         grid_bounds_.max.z - grid_bounds_.min.z);
  printf("VOLUME_GRID_STATS: scatter_mean=%.6f extinction_mean=%.6f extinction_max=%.6f "
         "occupied=%.4f\n",
         scatter_sum / double(voxel_count),
         extinct_sum / double(voxel_count),
         extinct_max,
         double(occupied) / double(voxel_count));
  fflush(stdout);
}

void VolumeModule::draw_prepass(View &main_view)
{
  if (!enabled_) {
    return;
  }

  GPU_debug_group_begin("Volumes");
  occupancy_fb_.bind();
  inst_.pipelines.world_volume.render(main_view);

  /* TODO(fclem): The infinite projection matrix makes the culling test unreliable (see #115595).
   * We need custom culling for these but that's not implemented yet. */
  volume_view.visibility_test(false);

  if (!current_objects_.is_empty()) {
    inst_.pipelines.volume.render(volume_view, occupancy_tx_);
  }
  draw_grid_prepass();
  GPU_debug_group_end();
}

void VolumeModule::draw_compute(View &main_view, int2 extent)
{
  if (!enabled_) {
    return;
  }

  if (inst_.pipelines.deferred.is_empty()) {
    /* This assume the volume are computed after deferred passes. This is needed to avoid broken
     * lighting and shadowing as the lights are not setup otherwise (see #121971). */
    inst_.hiz_buffer.swap_layer();
    inst_.hiz_buffer.update();
    inst_.volume_probes.set_view(main_view);
    inst_.sphere_probes.set_view(main_view);
    inst_.shadows.render(main_view, extent);
  }

  scatter_tx_.swap();
  extinction_tx_.swap();

  inst_.manager->submit(scatter_ps_, main_view);
  inst_.manager->submit(integration_ps_, main_view);

  /* Copy history data. */
  history_viewmat_ = main_view.viewmat();
  data_.history_depth_near = data_.depth_near;
  data_.history_depth_far = data_.depth_far;
  data_.history_depth_distribution = data_.depth_distribution;
  data_.history_winmat_stable = data_.winmat_stable;
  valid_history_ = true;
  history_frame_count_ += 1;
}

void VolumeModule::draw_resolve(View &view)
{
  if (!enabled_) {
    return;
  }

  inst_.hiz_buffer.update();

  resolve_fb_.ensure(GPU_ATTACHMENT_NONE,
                     GPU_ATTACHMENT_TEXTURE(inst_.render_buffers.combined_tx));
  resolve_fb_.bind();
  inst_.manager->submit(resolve_ps_, view);
}

}  // namespace blender::eevee
