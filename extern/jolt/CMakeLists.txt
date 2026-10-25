# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Jolt Physics (https://github.com/jrouwe/JoltPhysics), vendored at version 5.1.0.
#
# Jolt ships its own top level CMakeLists.txt (Build/CMakeLists.txt) which includes
# Jolt/Jolt.cmake. That file overrides global CXX flags, the MSVC runtime library and
# optimization settings, which would interfere with Blender's build. Instead the source
# files are compiled directly here, replicating only the compile definitions Jolt needs.
#
# The compile definitions below are PUBLIC on purpose: Jolt's headers select code paths and
# inline implementations based on them, so every translation unit that includes Jolt headers
# must use exactly the same set, otherwise the ABI of Jolt's inline functions would differ.

set(JOLT_PHYSICS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/Jolt)

file(GLOB_RECURSE JOLT_SRC_FILES CONFIGURE_DEPENDS ${JOLT_PHYSICS_ROOT}/*.cpp)

add_library(extern_jolt STATIC ${JOLT_SRC_FILES})

set_target_properties(extern_jolt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "extern"
)

# Jolt's sources use includes like "Jolt/Jolt.h", so the include directory is the parent of the
# Jolt source directory (equivalent to Jolt's own PHYSICS_REPO_ROOT).
target_include_directories(extern_jolt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(extern_jolt PUBLIC
  # Match the release configuration: asserts are disabled outside of Debug builds.
  $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
  JPH_OBJECT_STREAM
  JPH_OBJECT_LAYER_BITS=16
)

# Only enable instruction set extensions that are part of the x86-64 baseline plus the
# broadly available SSE4/LZCNT/TZCNT set. AVX/AVX2/FMA are deliberately not enabled here:
# those would require /arch:AVX2 on every translation unit that includes Jolt headers,
# including Blender's own code.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|x86|i[3-6]86)$")
  target_compile_definitions(extern_jolt PUBLIC
    JPH_USE_SSE4_1
    JPH_USE_SSE4_2
    JPH_USE_LZCNT
    JPH_USE_TZCNT
  )
endif()

if(MSVC)
  # Jolt does not use RTTI. Warnings are silenced: this is vendored third party code.
  target_compile_options(extern_jolt PRIVATE
    /GR-
    /Zc:__cplusplus
    /fp:except-
    /W0
  )
endif()

# Blender style alias, so other libraries can link against `bf::extern::jolt`.
add_library(bf::extern::jolt ALIAS extern_jolt)
