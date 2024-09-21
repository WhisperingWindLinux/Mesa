/*
 * Copyright 2024 Autodesk, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <stdio.h>

#include "util/macros.h"
#include "util/ralloc.h"
#include "util/disk_cache.h"

#include "nir.h"
#include "nir_builder.h"

#include "frontend/sw_winsys.h"
#include "gallium/winsys/sw/null/null_sw_winsys.h"

#include "pipe/p_screen.h"
#include "pipe/p_context.h"

#include "gallium/drivers/llvmpipe/lp_state_fs.h"
#include "gallium/auxiliary/gallivm/lp_bld_format.h"

#include "lp_context.h"
#include "lp_public.h"
#include "lp_screen.h"
#include "lp_test.h"

static const unsigned vec4_size = sizeof(float) * 4;

#define QUAD_LENGTH 2
#define QUAD_SIZE (QUAD_LENGTH * QUAD_LENGTH)
#define BLOCK_LENGTH (2 * QUAD_LENGTH)
#define BLOCK_SIZE (QUAD_SIZE * QUAD_SIZE)

#define COLOR_BUFFER_COUNT 1

static const unsigned quad_mask_location = 0;

static const unsigned data_buffer_location = 1;
#define DATA_BUFFER_SIZE 4
static const float data_buffer[DATA_BUFFER_SIZE][4] = {
   {3, 5, 11, 17},
   {2, 7, 11, 17},
   {2, 5, 13, 17},
   {2, 5, 11, 19},
};

static const unsigned image_length = 2;
static const unsigned image_size = image_length * image_length;

static const unsigned descriptor_set_location = 2;
static const unsigned texture_descriptor_set_index = 0;
static const unsigned image_descriptor_set_index = 1;
static const unsigned global_buffer_descriptor_set_index = 2;

static const float fs_inputs[2][4] = {{0, 0, 0, 1}, {0, 0, 7, 23}};
static const float fs_inputs_dx[2][4] = {{1, 0, 0, 0}, {1, 0, 6, 0}};
static const float fs_inputs_dy[2][4] = {{0, 1, 0, 0}, {0, 1, 0, 8}};

static const float unset_output_value[4] = {9999, 9999, 9999, 9999};

static const nir_shader_compiler_options shader_options = {
   .has_ddx_intrinsics = true,
   .scalarize_ddx = true,
};

static const float uniform_derivatives_quad_output[QUAD_SIZE][4] = {
   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
};
static const float indices_derivatives_quad_output[QUAD_SIZE][4] = {
   {1, 1, 6, 8}, {1, 1, 6, 8}, {1, 1, 6, 8}, {1, 1, 6, 8},
};
static const float data_derivatives_quad_output[QUAD_SIZE][4] = {
   {2, 2, 2, 0}, {1, 4, 0, 2}, {1, 0, 4, 2}, {0, 2, 2, 4},
};


enum test_variant {
   test_variant_rasterizer_mask,
   test_variant_terminated_mask,
   test_variant_demoted_mask,
   test_variant_diverged_mask,

   test_variant_count
};

static const char *test_variant_name[] = {
   "rasterizer_mask",
   "terminated_mask",
   "demoted_mask",
   "diverged_mask",
};


static bool
check_quad_output(const char* name,
                  enum test_variant variant,
                  bool is_uniform_access,
                  unsigned quad_mask,
                  const float actual_quad_output[QUAD_SIZE][4],
                  const float expected_quad_output[QUAD_SIZE][4])
{
   bool expected_equal;
   switch (variant)
   {
   case test_variant_rasterizer_mask:
      expected_equal = true;
      break;
   case test_variant_demoted_mask:
   /* llvmpipe bug, should be false */
   case test_variant_terminated_mask:
      expected_equal = true;
      break;
   case test_variant_diverged_mask:
      /* derivatives are correct only when no divergence */
      expected_equal = is_uniform_access || (quad_mask == 0xF);
      break;
   default:
      abort();
   }

   bool success = expected_equal;
   for (unsigned i = 0; i < QUAD_SIZE; i++) {
      const float *expected = quad_mask & (1 << i) ? expected_quad_output[i] : unset_output_value;
      const bool equal = memcmp(actual_quad_output[i], expected, vec4_size) == 0;
      if (expected_equal) {
         success &= equal;
      } else {
         success |= !equal;
      }
   }

   if (!success) {
      printf("Test %s (%s 0x%x)\n", name, test_variant_name[variant], quad_mask);

      printf(expected_equal ? "Expected:\n" : "Unexpected:\n");
      for (int i = 0; i < QUAD_SIZE; i++) {
         const float *expected = quad_mask & (1 << i) ? expected_quad_output[i] : unset_output_value;
         printf("   %f, %f, %f, %f\n", expected[0], expected[1], expected[2], expected[3]);
      }

      printf("Actual:\n");

      for (int i = 0; i < QUAD_SIZE; i++) {
         const float *actual = actual_quad_output[i];
         printf("   %f, %f, %f, %f\n", actual[0], actual[1], actual[2], actual[3]);
      }
      printf("\n");
   }

   return success;
}


static struct lp_texture_handle *
create_texture_handle(struct pipe_context *ctx)
{
   struct pipe_screen *pscreen = ctx->screen;

   struct pipe_resource resource_template = {};
   resource_template.screen = pscreen;
   resource_template.target = PIPE_TEXTURE_2D;
   resource_template.format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   resource_template.width0 = image_length;
   resource_template.height0 = image_length;
   resource_template.depth0 = 1;
   resource_template.array_size = 1;
   resource_template.bind |= PIPE_BIND_SAMPLER_VIEW;
   resource_template.bind |= PIPE_BIND_SHADER_IMAGE;
   resource_template.flags = PIPE_RESOURCE_FLAG_DONT_OVER_ALLOCATE;

   uint64_t size;
   struct pipe_resource *resource = pscreen->resource_create_unbacked(pscreen, &resource_template, &size);

   struct llvmpipe_memory_allocation alloc = {.cpu_addr = (void *)data_buffer};
   pscreen->resource_bind_backing(pscreen, resource, (struct pipe_memory_allocation *)&alloc, 0, 0, 0);

   struct pipe_sampler_view view_template = {};
   view_template.target = PIPE_TEXTURE_2D;
   view_template.swizzle_r = PIPE_SWIZZLE_X;
   view_template.swizzle_g = PIPE_SWIZZLE_Y;
   view_template.swizzle_b = PIPE_SWIZZLE_Z;
   view_template.swizzle_a = PIPE_SWIZZLE_W;
   view_template.format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   view_template.u.buf.size = image_size * vec4_size;
   view_template.texture = resource;
   view_template.context = ctx;
   struct pipe_sampler_view *view = ctx->create_sampler_view(ctx, resource, &view_template);

   struct pipe_sampler_state sampler = {};
   sampler.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;

   struct lp_texture_handle *handle = (struct lp_texture_handle *)(uintptr_t)ctx->create_texture_handle(ctx, view, &sampler);

   ctx->sampler_view_destroy(ctx, view);
   pipe_resource_reference(&resource, NULL);

   return handle;
}


static struct lp_texture_handle *
create_image_handle(struct pipe_context *ctx)
{
   struct pipe_screen *pscreen = ctx->screen;

   struct pipe_resource resource_template = {};
   resource_template.screen = pscreen;
   resource_template.target = PIPE_TEXTURE_2D;
   resource_template.format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   resource_template.width0 = image_length;
   resource_template.height0 = image_length;
   resource_template.depth0 = 1;
   resource_template.array_size = 1;
   resource_template.bind |= PIPE_BIND_SAMPLER_VIEW;
   resource_template.bind |= PIPE_BIND_SHADER_IMAGE;
   resource_template.flags = PIPE_RESOURCE_FLAG_DONT_OVER_ALLOCATE;

   uint64_t size;
   struct pipe_resource *resource = pscreen->resource_create_unbacked(pscreen, &resource_template, &size);

   struct llvmpipe_memory_allocation alloc = {.cpu_addr = (void *)data_buffer};
   pscreen->resource_bind_backing(pscreen, resource, (struct pipe_memory_allocation *)&alloc, 0, 0, 0);

   struct pipe_image_view view = {};
   view.resource = resource;
   view.format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   view.u.buf.size = image_size * vec4_size;
   view.access = PIPE_IMAGE_ACCESS_READ;
   view.shader_access = PIPE_IMAGE_ACCESS_READ;
   struct lp_texture_handle *handle = (struct lp_texture_handle *)(uintptr_t)ctx->create_image_handle(ctx, &view);

   pipe_resource_reference(&resource, NULL);

   return handle;
}


static bool
run_shader(struct pipe_context *ctx,
           const char* name,
           enum test_variant variant,
           bool is_uniform_access,
           nir_shader *shader,
           const float expected_quad_output[QUAD_SIZE][4])
{
   struct pipe_shader_state state = {};
   state.type = PIPE_SHADER_IR_NIR;
   state.ir.nir = shader;
   void *fs_state = ctx->create_fs_state(ctx, &state);
   ctx->bind_fs_state(ctx, fs_state);

   struct lp_fragment_shader *fs = (struct lp_fragment_shader *)llvmpipe_context(ctx)->fs;

   const unsigned key_size = sizeof(struct lp_fragment_shader_variant_key) + sizeof(struct lp_sampler_static_state) + sizeof(struct lp_image_static_state);
   fs->variant_key_size = key_size;
   struct lp_fragment_shader_variant_key *key = (struct lp_fragment_shader_variant_key *)alloca(key_size);
   memset(key, 0, key_size);

   key->blend.rt[0].colormask = PIPE_MASK_RGBA;
   key->nr_cbufs = 1;
   key->nr_samplers = 1;
   key->nr_images = 1;
   key->cbuf_format[0] = PIPE_FORMAT_R32G32B32A32_FLOAT;
   key->cbuf_nr_samples[0] = 1;
   key->coverage_samples = key->min_samples = key->no_ms_sample_mask_out = 1;

   struct lp_static_texture_state key_texture = {};
   key_texture.format = key_texture.res_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   key_texture.target = PIPE_TEXTURE_2D;
   key_texture.swizzle_r = PIPE_SWIZZLE_X;
   key_texture.swizzle_g = PIPE_SWIZZLE_Y;
   key_texture.swizzle_b = PIPE_SWIZZLE_Z;
   key_texture.swizzle_a = PIPE_SWIZZLE_W;
   key_texture.pot_width = key_texture.pot_height = key_texture.pot_depth = true;

   lp_fs_variant_key_images(key)[0].image_state = key_texture;
   struct lp_sampler_static_state *key_sampler = &lp_fs_variant_key_samplers(key)[0];
   key_sampler->texture_state = key_texture;
   key_sampler->sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;

   struct lp_fragment_shader_variant *fs_variant = lp_generate_variant(llvmpipe_context(ctx), fs, key);

   struct lp_jit_viewport viewport = {.min_depth = 0, .max_depth = 1};
   struct lp_jit_context jit_context = {};
   jit_context.viewports = &viewport;
   jit_context.sample_mask = ~0;

   unsigned quad_mask_buffer[1];

   struct lp_descriptor descriptor_sets[3] = {};

   struct lp_texture_handle *texture_handle = create_texture_handle(ctx);

   struct lp_jit_texture jit_texture = {};
   jit_texture.base = data_buffer;
   jit_texture.width = image_length;
   jit_texture.height = image_length;
   jit_texture.depth = 1;
   jit_texture.row_stride[0] = image_length * vec4_size;
   jit_texture.sampler_index = texture_handle->sampler_index;

   struct lp_jit_sampler jit_sampler = {};
   jit_sampler.max_aniso = 1;

   descriptor_sets[texture_descriptor_set_index].texture = jit_texture;
   descriptor_sets[texture_descriptor_set_index].sampler = jit_sampler;
   descriptor_sets[texture_descriptor_set_index].functions = texture_handle->functions;

   struct lp_texture_handle *image_handle = create_image_handle(ctx);

   struct lp_jit_image jit_image = {};
   jit_image.base = data_buffer;
   jit_image.width = image_length;
   jit_image.height = image_length;
   jit_texture.depth = 1;
   jit_image.row_stride = image_length * vec4_size;

   descriptor_sets[image_descriptor_set_index].image = jit_image;
   descriptor_sets[image_descriptor_set_index].functions = image_handle->functions;

   const float *global_buffer_ptr[1] = {(const float *)data_buffer};
   descriptor_sets[global_buffer_descriptor_set_index].buffer.u = (const void *)global_buffer_ptr;
   descriptor_sets[global_buffer_descriptor_set_index].buffer.num_elements = sizeof(float*);

   struct lp_jit_resources jit_resources = {};
   jit_resources.constants[quad_mask_location].u = (const void *)quad_mask_buffer;
   jit_resources.constants[quad_mask_location].num_elements = sizeof(quad_mask_buffer);
   jit_resources.constants[data_buffer_location].u = (const void *)data_buffer;
   jit_resources.constants[data_buffer_location].num_elements = DATA_BUFFER_SIZE * vec4_size;
   jit_resources.constants[descriptor_set_location].u = (const void *)descriptor_sets;
   jit_resources.constants[descriptor_set_location].num_elements = ARRAY_SIZE(descriptor_sets);
   jit_resources.ssbos[data_buffer_location].u = (const void *)data_buffer;
   jit_resources.ssbos[data_buffer_location].num_elements = DATA_BUFFER_SIZE * vec4_size;
   jit_resources.aniso_filter_table = lp_build_sample_aniso_filter_table();

   struct lp_jit_thread_data thread_data = {};
   struct lp_build_format_cache thread_cache = {};
   thread_data.cache = &thread_cache;

   unsigned color_stride[COLOR_BUFFER_COUNT] = {QUAD_SIZE * vec4_size};
   unsigned color_sample_stride[COLOR_BUFFER_COUNT] = {BLOCK_SIZE * vec4_size};

   uint8_t *depth_buffer = NULL;
   unsigned depth_stride = 0;
   unsigned depth_sample_stride = 0;

   float color_buffer[BLOCK_SIZE][4];
   uint8_t *color_buffers[1] = {(uint8_t*)color_buffer};

   float quad_output[QUAD_SIZE][4];

   bool success = true;
   for (unsigned i = 0; i < QUAD_SIZE; i++) {
      unsigned quad_mask = (1 << (i + 1)) - 1;

      unsigned block_mask = 0x33;
      if (variant == test_variant_rasterizer_mask) {
         block_mask = (quad_mask & 0x3) | (quad_mask & 0xC) << 2;
      }

      quad_mask_buffer[0] = quad_mask;

      for (int i = 0; i < BLOCK_SIZE; i++) {
         memcpy(color_buffer[i], unset_output_value, vec4_size);
      }

      fs_variant->jit_function[RAST_EDGE_TEST](&jit_context,
                                             &jit_resources,
                                             0, 0, 1,
                                             fs_inputs,
                                             fs_inputs_dx,
                                             fs_inputs_dy,
                                             color_buffers,
                                             depth_buffer,
                                             block_mask,
                                             &thread_data,
                                             color_stride,
                                             depth_stride,
                                             color_sample_stride,
                                             depth_sample_stride);

      memcpy(quad_output, color_buffer, QUAD_LENGTH * vec4_size);
      memcpy(quad_output + QUAD_LENGTH, color_buffer + BLOCK_LENGTH, QUAD_LENGTH * vec4_size);
      success &= check_quad_output(name, variant, is_uniform_access, quad_mask, quad_output, expected_quad_output);
   }

   if (!success) {
      nir_print_shader(shader, stdout);
      printf("\n\n");
   }

   /* will also delete nir shader and variants */
   ctx->delete_fs_state(ctx, fs_state);

   ctx->delete_texture_handle(ctx, (uint64_t)(uintptr_t)texture_handle);
   ctx->delete_image_handle(ctx, (uint64_t)(uintptr_t)image_handle);

   return success;
}

#define CHECK_QUAD_OUTPUT(expected_quad_output) \
   return run_shader(ctx, __FUNCTION__, variant, false, shader, expected_quad_output);

#define CHECK_QUAD_OUTPUT_UNIFORM(expected_quad_output) \
   return run_shader(ctx, __FUNCTION__, variant, true, shader, expected_quad_output);


static nir_def*
nir_fwidth(nir_builder *b, nir_def* src)
{
   return nir_fadd(b,
            nir_fabs(b, nir_ddx(b, src)),
            nir_fabs(b, nir_ddy(b, src)));
}


static nir_def *
indices_to_index(nir_builder *b, struct nir_variable *indices_var)
{
   nir_def *indices = nir_f2i32(b, nir_load_var(b, indices_var));
   return nir_iadd(b, nir_channel(b, indices, 0),
         nir_imul_imm(b, nir_channel(b, indices, 1), 2));
}

static nir_def *
indices_to_offset(nir_builder *b, struct nir_variable *indices_var,
      unsigned num_components, unsigned bit_size)
{
   nir_def *index = indices_to_index(b, indices_var);
   return nir_imul_imm(b, index, num_components * (bit_size / 8));
}


struct shader_vars {
   nir_variable *in_indices;
   nir_variable *out_value;
   nir_variable *last_frag_data;

   nir_if *mask_check;
};

static nir_def *
index_mask_check(nir_builder *b, struct shader_vars *vars)
{
   nir_def *quad_index = indices_to_index(b, vars->in_indices);
   nir_def *mask = nir_load_ubo(b, 1, 32, nir_imm_int(b, quad_mask_location), nir_imm_int(b, 0), .range = ~0);
   nir_def *mask_bit = nir_iand(b, mask, nir_ishl(b, nir_imm_int(b, 1), quad_index));
   return nir_ine32(b, mask_bit, nir_imm_int(b, 0));
}

static struct shader_vars
start_shader(nir_builder *b, enum test_variant variant)
{
   *b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &shader_options, "lp_test_helper_invocation");

   struct shader_vars vars = {};
   vars.in_indices = nir_create_variable_with_location(b->shader, nir_var_shader_in, VARYING_SLOT_VAR0, &glsl_type_builtin_vec4);
   vars.out_value = nir_create_variable_with_location(b->shader, nir_var_shader_out, FRAG_RESULT_DATA0, &glsl_type_builtin_vec4);
   vars.last_frag_data = nir_create_variable_with_location(b->shader, nir_var_shader_out, FRAG_RESULT_DATA0, &glsl_type_builtin_vec4);
   vars.last_frag_data->data.fb_fetch_output = 1;

   switch (variant)
   {
   case test_variant_rasterizer_mask:
      break;
   case test_variant_terminated_mask:
   case test_variant_demoted_mask: {
      /* TODO: llvmpipe implements demotion semantics with terminate,
         and lowers "demote" intrinsics to "terminate",
         but it should be the other way around!
         Although in reality in should have both, because
         both behaviours are required for Vulkan. */
      nir_terminate_if(b, nir_inot(b, index_mask_check(b, &vars)));
   } break;
   case test_variant_diverged_mask:
      vars.mask_check = nir_push_if(b, index_mask_check(b, &vars));
      break;
   default:
      abort();
   }

   return vars;
}


static nir_shader *
end_shader(nir_builder *b, struct shader_vars *vars, nir_def *data)
{
   nir_def *data_fwidth = nir_fwidth(b, data);

   if (vars->mask_check) {
      nir_push_else(b, vars->mask_check);
      nir_def *data_unset = nir_imm_vec4(b,
                                         unset_output_value[0],
                                         unset_output_value[1],
                                         unset_output_value[2],
                                         unset_output_value[3]);
      nir_pop_if(b, vars->mask_check);
      data_fwidth = nir_if_phi(b, data_fwidth, data_unset);
   }

   nir_deref_instr *frag = nir_build_deref_var(b, vars->out_value);
   nir_store_deref(b, frag, data_fwidth, 0xf);

   nir_validate_shader(b->shader, NULL);
   return b->shader;
}


static bool
test_load_input_var(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_load_deref(b, nir_build_deref_var(b, vars.in_indices));
      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT_UNIFORM(indices_derivatives_quad_output);
}


static bool
test_load_output_var(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_load_deref(b, nir_build_deref_var(b, vars.in_indices));
      data = nir_fadd_imm(b, data, 11); /* Won't change the derivative */
      nir_store_deref(b, nir_build_deref_var(b, vars.out_value), data, 0xf);

      data = nir_load_deref(b, nir_build_deref_var(b, vars.out_value));
      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT(indices_derivatives_quad_output);
}


static bool
test_fetch_framebuffer(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_load_deref(b, nir_build_deref_var(b, vars.last_frag_data));
      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool
test_load_reg(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *reg = nir_decl_reg(b, 4, 32, 0);
      nir_build_store_reg(b, nir_imm_vec4(b, 5, 5, 11, 17), reg);

      nir_def *data = nir_load_reg(b, reg);
      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT(uniform_derivatives_quad_output);
}


static bool
test_load_reg_indirect(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *reg = nir_decl_reg(b, 4, 32, DATA_BUFFER_SIZE);
      for (unsigned row_index = 0; row_index < DATA_BUFFER_SIZE; row_index++) {
         const float *row = data_buffer[row_index];
         nir_build_store_reg(b, nir_imm_vec4(b, row[0], row[1], row[2], row[3]),
               reg, .base = row_index);
      }

      nir_def *index = indices_to_index(b, vars.in_indices);
      nir_def *data = nir_load_reg_indirect(b, 4, 32, reg, index);
      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT(data_derivatives_quad_output);
}


static bool
test_load_ubo_uniform(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_load_ubo(b, 4, 32, nir_imm_int(b, data_buffer_location), nir_imm_int(b, 0), .range = ~0);

      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}

static bool
test_load_ubo(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *offset = indices_to_offset(b, vars.in_indices, 4, 32);
      nir_def *data = nir_load_ubo(b, 4, 32, nir_imm_int(b, data_buffer_location), offset, .range = ~0);

      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT(data_derivatives_quad_output);
}


static bool
test_load_global(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *address = nir_load_ubo(b, 1, 64, nir_imm_ivec2(b, descriptor_set_location, global_buffer_descriptor_set_index), nir_imm_int(b, 0), .range = ~0);
      address = nir_iadd(b, address, nir_i2i64(b, indices_to_offset(b, vars.in_indices, 4, 32)));
      nir_def *data = nir_load_global(b, address, 16, 4, 32);

      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT(data_derivatives_quad_output);
}


static bool
test_load_ssbo_uniform(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_load_ssbo(b, 4, 32, nir_imm_int(b, data_buffer_location), nir_imm_int(b, 0));

      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool
test_load_ssbo(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *offset = indices_to_offset(b, vars.in_indices, 4, 32);
      nir_def *data = nir_load_ssbo(b, 4, 32, nir_imm_int(b, data_buffer_location), offset);

      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT(data_derivatives_quad_output);
}

static bool
test_load_ssbo_size(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder* b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_i2f32(b, nir_get_ssbo_size(b, nir_imm_int(b, data_buffer_location)));
      shader = end_shader(b, &vars, nir_pad_vector_imm_int(b, data, 0, 4));
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static nir_def *
nir_tex_deref_handle(nir_builder *b,
                     nir_def *handle,
                     nir_def *coord)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_tex;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   nir_def_init(&tex->instr, &tex->def, nir_tex_instr_dest_size(tex), 32);
   nir_builder_instr_insert(b, &tex->instr);

   return &tex->def;
}

static bool
test_tex_uniform(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_tex_deref_handle(b, nir_imm_ivec3(b, descriptor_set_location, texture_descriptor_set_index, 0),
            nir_imm_ivec2(b, 0, 0));
      shader = end_shader(b, &vars, data);
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool
test_tex(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *coord = nir_trim_vector(b, nir_load_var(b, vars.in_indices), 2);
      coord = nir_fdiv_imm(b, coord, image_length);
      nir_def *data = nir_tex_deref_handle(b, nir_imm_ivec3(b, descriptor_set_location, texture_descriptor_set_index, 0), coord);
      shader = end_shader(b, &vars, data);
   }

   /* Sampling is always done in quad groups, so all invocations fetch if one fetches. */
   CHECK_QUAD_OUTPUT_UNIFORM(data_derivatives_quad_output);
}


static nir_def *
nir_tex_size_handle(nir_builder *b,
                     nir_def *handle)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txs;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_int32;
   nir_def_init(&tex->instr, &tex->def, nir_tex_instr_dest_size(tex), 32);
   nir_builder_instr_insert(b, &tex->instr);

   return &tex->def;
}


static bool
test_tex_size(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_tex_size_handle(b, nir_imm_ivec3(b, descriptor_set_location, texture_descriptor_set_index, 0));
      data = nir_i2f32(b, data);
      shader = end_shader(b, &vars, nir_pad_vector_imm_int(b, data, 0, 4));
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool
test_sysval_intrin(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_load_sample_pos(b);
      shader = end_shader(b, &vars, nir_pad_vector_imm_int(b, data, 0, 4));
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool
test_image_op_uniform(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_bindless_image_load(b, 4, 32, nir_imm_ivec3(b, descriptor_set_location, image_descriptor_set_index, 0),
         nir_imm_ivec4(b, 0, 0, 0, 0), nir_imm_int(b, 0), nir_imm_int(b, 0),
         .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32G32B32A32_FLOAT);
      shader = end_shader(b, &vars, nir_pad_vector_imm_int(b, data, 0, 4));
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool
test_image_op(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *coord = nir_f2i32(b, nir_load_var(b, vars.in_indices));
      nir_def *data = nir_bindless_image_load(b, 4, 32, nir_imm_ivec3(b, descriptor_set_location, image_descriptor_set_index, 0),
         coord, nir_imm_int(b, 0), nir_imm_int(b, 0),
         .image_dim = GLSL_SAMPLER_DIM_1D, .format = PIPE_FORMAT_R32G32B32A32_FLOAT);
      shader = end_shader(b, &vars, nir_pad_vector_imm_int(b, data, 0, 4));
   }

   /* Sampling is always done in quad groups, so all invocations fetch if one fetches. */
   CHECK_QUAD_OUTPUT_UNIFORM(data_derivatives_quad_output);
}


static bool
test_image_size(unsigned verbose, FILE *fp,
               struct pipe_context *ctx,
               enum test_variant variant)
{
   nir_shader *shader;
   {
      nir_builder bld;
      nir_builder *b = &bld;
      struct shader_vars vars = start_shader(b, variant);

      nir_def *data = nir_bindless_image_size(b, 2, 32, nir_imm_ivec3(b, descriptor_set_location, image_descriptor_set_index, 0),
         nir_imm_int(b, 0), .image_dim = GLSL_SAMPLER_DIM_2D, .format = PIPE_FORMAT_R32G32B32A32_FLOAT);
      data = nir_i2f32(b, data);
      shader = end_shader(b, &vars, nir_pad_vector_imm_int(b, data, 0, 4));
   }

   CHECK_QUAD_OUTPUT_UNIFORM(uniform_derivatives_quad_output);
}


static bool (*const test_cases[])(unsigned, FILE *, struct pipe_context *, enum test_variant) = {
   test_load_input_var,
   test_load_output_var,
   test_fetch_framebuffer,
   test_load_reg,
   test_load_reg_indirect,
   test_load_ubo_uniform,
   test_load_ubo,
    /* No uniform code path in fragment shaders,
       see invocation_0_must_be_active */
   test_load_global,
   test_tex_uniform,
   test_tex,
   test_tex_size,
   test_sysval_intrin,
   test_load_ssbo_uniform,
   test_load_ssbo,
   test_load_ssbo_size,
   test_image_op_uniform,
   test_image_op,
   test_image_size,
};


bool
test_all(unsigned verbose, FILE *fp)
{
   setenv("MESA_SHADER_CACHE_DISABLE", "true", 1);

   glsl_type_singleton_init_or_ref();

   struct sw_winsys *winsys = null_sw_create();
   struct pipe_screen *screen = llvmpipe_create_screen(winsys);
   struct pipe_context *ctx = screen->context_create(screen, NULL, 0);

   bool result = true;
   for (int variant = 0; variant < test_variant_count; variant++) {
      for (int i = 0; i < ARRAY_SIZE(test_cases); i++) {
         result &= test_cases[i](verbose, fp, ctx, variant);
      }
   }

   ctx->destroy(ctx);
   screen->destroy(screen);
   winsys->destroy(winsys);

   glsl_type_singleton_decref();

   return result;
}


bool
test_some(unsigned verbose, FILE *fp,
          unsigned long n)
{
   return test_all(verbose, fp);
}


bool
test_single(unsigned verbose, FILE *fp)
{
   printf("no test_single()");
   return true;
}

void
write_tsv_header(FILE *fp)
{
   fprintf(fp,
           "result\t"
           "format\n");

   fflush(fp);
}
