# Copyright © 2017-2018 Intel Corporation
# SPDX-License-Identifier: MIT

# TODO: support non-static targets
# Static targets are always enabled in autotools (unless you modify
# configure.ac)

omx_link_args = []
omx_link_depends = []

omx_sym = configure_file(input : 'omx.sym.in', output : 'omx.sym', configuration : sym_config)

if with_ld_version_script
  omx_link_args += ['-Wl,--version-script', join_paths(meson.current_build_dir(), 'omx.sym')]
  omx_link_depends += omx_sym
endif

link_with_libomx_gallium = []
if with_dri
  link_with_libomx_gallium += libgallium_dri
else
  if with_platform_windows
    link_with_libomx_gallium += libgallium_wgl
  endif
endif

libomx_gallium = shared_library(
  'omx_mesa',
  'omx_target.c',
  gnu_symbol_visibility : 'hidden',
  link_args : [omx_link_args, ld_args_gc_sections],
  include_directories : [
    inc_include, inc_src, inc_mapi, inc_mesa, inc_gallium, inc_gallium_aux, inc_util, inc_gallium_winsys, inc_gallium_drivers,
  ],
  link_whole : [libomx_st],
  link_with : [
    libgalliumvlwinsys, libgalliumvl, link_with_libomx_gallium,
    libpipe_loader_static, libws_null, libwsw, libswdri, libswkmsdri,
  ],
  link_depends : omx_link_depends,
  dependencies : [idep_mesautil],
  install : true,
  install_dir : omx_drivers_path,
)
