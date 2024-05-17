/*
 * Copyright © 2023 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"

/* Try to split a temporary variable which is a struct into its parts. This
 * can allow us to remove unused struct members, shrinking its size.
 */

struct var_node {
   nir_variable *children[0];
};

struct split_vars_state {
   void *dead_ctx;
   nir_shader *shader;
   struct hash_table *vars;
};

static void
register_var(struct split_vars_state *state, nir_variable *var)
{
   if (!glsl_type_is_struct(var->type))
      return;

   unsigned num_fields = glsl_get_length(var->type);
   size_t size = sizeof(struct var_node) +
      num_fields * sizeof(nir_variable *);
   struct var_node *node = rzalloc_size(state->dead_ctx, size);
   _mesa_hash_table_insert(state->vars, var, node);
}

static bool
create_var_children(struct split_vars_state *state, nir_variable *var,
                    nir_function_impl *impl)
{
   struct hash_entry *var_entry =
      _mesa_hash_table_search(state->vars, var);

   if (!var_entry)
      return false;

   struct var_node *node = var_entry->data;

   for (unsigned i = 0; i < glsl_get_length(var->type); i++) {
      const struct glsl_type *child_type =
         var->type->fields.structure[i].type;

      char *name = NULL;
      if (var->name || var->type->fields.structure[i].name) {
         name = ralloc_asprintf(state->dead_ctx, "%s.%s",
                                var->name ? var->name : "(unnamed)",
                                var->type->fields.structure[i].name ?
                                var->type->fields.structure[i].name :
                                "(unnamed)");
      }

      if (impl) {
         node->children[i] =
            nir_local_variable_create(impl, child_type, name);
      } else {
         node->children[i] =
            nir_variable_create(state->shader, nir_var_mem_global, child_type, name);
      }
   }

   return true;
}

static bool
instr_filter(const nir_instr *instr, const void *_state)
{
   return instr->type == nir_instr_type_deref &&
      nir_instr_as_deref(instr)->deref_type == nir_deref_type_struct;
}

static nir_def *
lower_instr(nir_builder *b, nir_instr *instr, void *_state)
{
   struct split_vars_state *state = _state;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   nir_deref_instr *parent =
      nir_instr_as_deref(deref->parent.ssa->parent_instr);

   if (parent->deref_type != nir_deref_type_var)
      return NULL;

   struct hash_entry *entry =
      _mesa_hash_table_search(state->vars, parent->var);

   if (!entry)
      return NULL;

   struct var_node *node = entry->data;
   nir_variable *new_var = node->children[deref->strct.index];
   return &nir_build_deref_var(b, new_var)->def;
}

bool
nir_opt_split_vars(nir_shader *shader)
{
   struct split_vars_state state = { NULL };
   bool progress = false;

   state.dead_ctx = ralloc_context(NULL);
   state.shader = shader;
   state.vars = _mesa_pointer_hash_table_create(state.dead_ctx);

   nir_foreach_variable_with_modes (var, shader, nir_var_shader_temp) {
      register_var(&state, var);
   }

   nir_foreach_function_impl (impl, shader) {
      nir_foreach_variable_in_list (var, &impl->locals) {
         register_var(&state, var);
      }

      /* Remove variables which have complex uses */
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;

            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var)
               continue;

            struct hash_entry *var_entry =
               _mesa_hash_table_search(state.vars, deref->var);
            if (!var_entry)
               continue;

            if (nir_deref_instr_has_complex_use(deref, 0))
               _mesa_hash_table_remove(state.vars, var_entry);
         }
      }

      nir_foreach_variable_in_list (var, &impl->locals) {
         progress |= create_var_children(&state, var, impl);
      }
   }

   nir_foreach_variable_with_modes (var, shader, nir_var_shader_temp) {
      progress |= create_var_children(&state, var, NULL);
   }

   nir_foreach_function_impl (impl, shader) {
      progress |= nir_shader_lower_instructions(shader, instr_filter,
                                                lower_instr, &state);

      /* Delete the original variables. */
      nir_foreach_variable_in_list_safe (var, &impl->locals) {
         struct hash_entry *entry =
            _mesa_hash_table_search(state.vars, var);

         if (entry)
            exec_node_remove(&var->node);
      }
   }

   nir_foreach_variable_with_modes_safe (var, shader, nir_var_shader_temp) {
      struct hash_entry *entry =
         _mesa_hash_table_search(state.vars, var);

      if (entry)
         exec_node_remove(&var->node);
   }


   ralloc_free(state.dead_ctx);

   return progress;
}

