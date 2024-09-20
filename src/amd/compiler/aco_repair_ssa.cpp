/*
 * Copyright © 2024 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "aco_ir.h"

#include <functional>
#include <stack>

namespace aco {

namespace {

struct repair_state {
   Program* program;
   Block* block;
   std::unique_ptr<uint32_t[]> def_blocks;
   std::unordered_map<uint64_t, uint32_t> renames;

   std::vector<aco_ptr<Instruction>> new_phis;

   std::vector<bool> needs_temp;
   std::vector<uint32_t> temps;
};

Temp
create_phis(repair_state* state, Temp tmp, uint32_t use_block, uint32_t def_block)
{
   Program* program = state->program;

   assert(program->blocks[def_block].logical_idom != -1);
   assert(program->blocks[use_block].logical_idom != -1);

   std::fill(state->needs_temp.begin(), state->needs_temp.end(), false);

   state->needs_temp[use_block] = true;
   for (int32_t i = use_block - 1; i >= (int)def_block; i--) {
      bool block_needs_tmp = false;
      for (uint32_t succ : program->blocks[i].logical_succs)
         block_needs_tmp |= succ > (uint32_t)i && state->needs_temp[succ];
      state->needs_temp[i] = block_needs_tmp;
   }

   state->temps[def_block] = tmp.id();
   for (uint32_t i = def_block + 1; i <= use_block; i++) {
      if (!state->needs_temp[i])
         continue;

      bool undef = true;
      for (unsigned pred : program->blocks[i].logical_preds)
         undef &= pred < i && (!state->needs_temp[pred] || !state->temps[pred]);
      if (undef) {
         state->temps[i] = 0;
         continue;
      }

      /* If the immediate dominator has a temporary, we don't need to create a phi and can just use
       * that temporary instead. For linear temporaries, we also need to check if dominates in the
       * linear CFG, because logical dominators do not necessarily dominate a block in the linear
       * CFG (for example, because of continue_or_break). */
      Block& block = program->blocks[i];
      uint32_t idom = block.logical_idom;
      if (state->needs_temp[idom] && state->temps[idom] &&
          (!tmp.is_linear() || dominates_linear(program->blocks[idom], block))) {
         state->temps[i] = state->temps[idom];
         continue;
      }

      uint64_t k = block.index | ((uint64_t)tmp.id() << 32);
      auto it = state->renames.find(k);
      if (it != state->renames.end()) {
         state->temps[i] = it->second;
         continue;
      }

      /* This pass doesn't support creating loop header phis */
      assert(!(block.kind & block_kind_loop_header));

      Temp def = program->allocateTmp(tmp.regClass());
      aco_ptr<Instruction> phi{
         create_instruction(aco_opcode::p_phi, Format::PSEUDO, block.logical_preds.size(), 1)};
      for (unsigned j = 0; j < block.logical_preds.size(); j++) {
         uint32_t pred = block.logical_preds[j];
         phi->operands[j] =
            Operand(Temp(state->needs_temp[pred] ? state->temps[pred] : 0, tmp.regClass()));
      }
      phi->definitions[0] = Definition(def);

      /* Require all operands are defined to avoid fixing broken IR. */
      if ((debug_flags & DEBUG_VALIDATE_IR) && !(block.kind & block_kind_allow_repair_phis)) {
         if (std::any_of(phi->operands.begin(), phi->operands.end(),
                         std::mem_fn(&Operand::isUndefined))) {
            aco_err(state->program,
                    "Repair phi with undefined operands necessary at BB%u for %%%u (defined at "
                    "BB%u and used at BB%u)",
                    block.index, tmp.id(), def_block, use_block);
            assert(false);
         }
      }

      if (&block == state->block)
         state->new_phis.emplace_back(std::move(phi));
      else
         block.instructions.emplace(block.instructions.begin(), std::move(phi));

      state->renames.emplace(k, def.id());
      state->temps[i] = def.id();
   }

   return Temp(state->temps[use_block], tmp.regClass());
}

template <bool LoopHeader>
void
repair_block(repair_state* state, Block& block)
{
   state->block = &block;
   for (aco_ptr<Instruction>& instr : block.instructions) {
      for (Definition def : instr->definitions) {
         if (def.isTemp())
            state->def_blocks[def.tempId()] = block.index;
      }

      unsigned start = 0;
      unsigned num_operands = instr->operands.size();
      if ((is_phi(instr) || instr->opcode == aco_opcode::p_boolean_phi) &&
          (block.kind & block_kind_loop_header)) {
         if (LoopHeader)
            start++;
         else
            num_operands = 1;
      } else if (LoopHeader) {
         break;
      }

      for (unsigned i = start; i < num_operands; i++) {
         Operand& op = instr->operands[i];
         if (!op.isTemp())
            continue;

         uint32_t use_block = block.index;
         if (instr->opcode == aco_opcode::p_boolean_phi || instr->opcode == aco_opcode::p_phi)
            use_block = block.logical_preds[i];
         else if (instr->opcode == aco_opcode::p_linear_phi)
            use_block = block.linear_preds[i];

         uint32_t def_block = state->def_blocks[op.tempId()];
         bool dominates = op.getTemp().is_linear()
                             ? dominates_linear(state->program->blocks[def_block],
                                                state->program->blocks[use_block])
                             : dominates_logical(state->program->blocks[def_block],
                                                 state->program->blocks[use_block]);
         if (!dominates)
            op.setTemp(create_phis(state, op.getTemp(), use_block, def_block));
      }
   }

   /* These are inserted later to not invalidate any iterators. */
   block.instructions.insert(block.instructions.begin(),
                             std::move_iterator(state->new_phis.begin()),
                             std::move_iterator(state->new_phis.end()));
   state->new_phis.clear();
}

} /* end namespace */

void
repair_ssa(Program* program)
{
   repair_state state;
   state.program = program;
   state.def_blocks.reset(new uint32_t[program->peekAllocationId()]);

   state.needs_temp.resize(program->blocks.size());
   state.temps.resize(program->blocks.size());

   std::stack<unsigned, std::vector<unsigned>> loop_header_indices;

   for (Block& block : program->blocks) {
      if (block.kind & block_kind_loop_header)
         loop_header_indices.push(block.index);

      repair_block<false>(&state, block);

      if (block.kind & block_kind_loop_exit) {
         unsigned header = loop_header_indices.top();
         loop_header_indices.pop();

         repair_block<true>(&state, program->blocks[header]);
      }
   }
}

} // namespace aco
