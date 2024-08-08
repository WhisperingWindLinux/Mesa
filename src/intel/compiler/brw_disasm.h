/*
 * Copyright 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef BRW_DISASM_H
#define BRW_DISASM_H

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct brw_isa_info;
struct brw_inst;

const struct brw_label *brw_find_label(const struct brw_label *root, int offset);
void brw_create_label(struct brw_label **labels, int offset, void *mem_ctx);
int brw_disassemble_inst(FILE *file, const struct brw_isa_info *isa,
                         const struct brw_inst *inst, bool is_compacted,
                         int offset, const struct brw_label *root_label);
const struct
brw_label *brw_label_assembly(const struct brw_isa_info *isa,
                              const void *assembly, int start, int end,
                              void *mem_ctx);
void brw_disassemble_with_labels(const struct brw_isa_info *isa,
                                 const void *assembly, int start, int end, FILE *out);
void brw_disassemble(const struct brw_isa_info *isa,
                     const void *assembly, int start, int end,
                     const struct brw_label *root_label,
                     int64_t *lineno_offset, FILE *out);
int brw_disassemble_find_end(const struct brw_isa_info *isa,
                             const void *assembly, int start);
void brw_disassemble_with_errors(const struct brw_isa_info *isa,
                                 const void *assembly, int start,
                                 int64_t *lineno_offset, FILE *out);
void brw_disassemble_with_lineno(const struct brw_isa_info *isa, uint32_t stage,
                                 uint32_t dispatch_width, uint32_t src_hash,
                                 const void *assembly, int start,
                                 int64_t lineno_offset, FILE *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BRW_DISASM_H */
