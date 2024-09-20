# Copyright 2024 Advanced Micro Devices, Inc.
# All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#
# Generate dlopen/dlsym code for libdrm_amdgpu symbols we need.
# Usage:
# $ python3 libdrm_amdgpu_gen_loader.py /path/to/libdrm_amdgpu/amdgpu.h header > libdrm_amdgpu_loader.h
# $ python3 libdrm_amdgpu_gen_loader.py /path/to/libdrm_amdgpu/amdgpu.h source > libdrm_amdgpu_loader.c

import sys
from pycparser import c_ast, parse_file, c_generator

generator = c_generator.CGenerator()
# A simple visitor for FuncDef nodes that prints the names and
# locations of function definitions.
class FuncDefVisitor(c_ast.NodeVisitor):
    def __init__(self, header):
        self.types = []
        self.header = header


    def visit_FuncDecl(self, node):
        return_type = generator._generate_type(node.type.type, emit_declname=False)

        if isinstance(node.type, c_ast.PtrDecl):
            ctype = '{}_type'.format(node.type.type.declname)
            if self.header:
                print('typedef {}* (*{})'.format(return_type, ctype), end='')
        else:
            ctype = '{}_type'.format(node.type.declname)
            if self.header:
                print('typedef {} (*{})'.format(return_type, ctype), end='')

        self.types += [ctype]
        args = []
        for p in node.args.params:
            if p.name is None:
                continue
            args += [generator._generate_decl(p)]
        if self.header:
            print('({});'.format(', '.join(args)))

class FuncStubVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.types = []

    def visit_FuncDecl(self, node):
        return_type = generator._generate_type(node.type.type, emit_declname=False)

        if isinstance(node.type, c_ast.PtrDecl):
            print('static {}* {}_stub'.format(return_type, node.type.type.declname), end='')
        else:
            print('static {} {}_stub'.format(return_type, node.type.declname), end='')

        args = []
        for p in node.args.params:
            if p.name is None:
                continue
            args += [generator._generate_decl(p)]
        # args = [generator._generate_decl(p) for p in node.args.params]

        print('({})'.format(', '.join(args)), end='')

        print('''
            {
            printf("IMPLEMENT ME %s\\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        ''', end='')

        s = '{}'.format(return_type)
        if s == 'int ' or s == 'uint64_t ':
            print(' return -1; }')
        elif s == 'void ':
            print('}')
        else:
            print('return NULL; }')

def generate_header(filename):
    # Note that cpp is used. Provide a path to your own cpp or
    # make sure one exists in PATH.

    ast = parse_file(filename, use_cpp=True)
    print('''/*
 * Generated using libdrm_amdgpu_gen_loader.py script
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef AC_LIBDRM_AMDGPU_LOADER_H
#define AC_LIBDRM_AMDGPU_LOADER_H
#include <amdgpu.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
        ''')

    # Generate typedefs
    v = FuncDefVisitor(True)
    v.visit(ast)

    # Generate struct
    print('struct libdrm_amdgpu {')
    print('   void *handle;')
    print('   int32_t refcount;')
    for ty in v.types:
        print('{} {};'.format(ty, ty[len('amdgpu_'):-len('_type')]))
    print('};')

    print('''
struct libdrm_amdgpu * ac_init_libdrm_amdgpu(void);

void ac_deinit_libdrm_amdgpu(struct libdrm_amdgpu *h);

struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio(void);

struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio_stubs(void);

#ifdef __cplusplus
}
#endif

#endif /* AC_LIBDRM_AMDGPU_LOADER_H */
''')

def generate_source(filename):
    # Note that cpp is used. Provide a path to your own cpp or
    # make sure one exists in PATH.

    ast = parse_file(filename, use_cpp=True)
    print('''
#include "libdrm_amdgpu_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void ac_deinit_libdrm_amdgpu(struct libdrm_amdgpu *h) {
    if (h->handle)
        dlclose(h->handle);
    free(h);
}
struct libdrm_amdgpu * ac_init_libdrm_amdgpu(void) {
   struct libdrm_amdgpu *libdrm_amdgpu = calloc(1, sizeof(struct libdrm_amdgpu));
   void *libdrm = dlopen("libdrm_amdgpu.so.1", RTLD_NOW | RTLD_LOCAL);
   assert(libdrm);
        ''')

    # Generate typedefs
    v = FuncDefVisitor(False)
    v.visit(ast)

    # Generate struct
    for ty in v.types:
        local = ty[len('amdgpu_'):-len('_type')]
        original = ty[:-len('_type')]
        print('libdrm_amdgpu->{} = dlsym(libdrm, "{}");'.format(local, original))
        print('assert(libdrm_amdgpu->{} != NULL);'.format(local))

    print('libdrm_amdgpu->handle = libdrm;')
    print('''
   return libdrm_amdgpu;
}
''')
    FuncStubVisitor().visit(ast)

    print('''
    struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio_stubs(void) {
       struct libdrm_amdgpu *libdrm_amdgpu = calloc(1, sizeof(struct libdrm_amdgpu));
''')
    for ty in v.types:
        local = ty[len('amdgpu_'):-len('_type')]
        stub = '{}_stub'.format(ty[:-len('_type')])
        print('libdrm_amdgpu->{} = {};'.format(local, stub))

    print('''
   return libdrm_amdgpu;
}
''')

if __name__ == "__main__":
    # TODO: generate default stubs function for virtio. This way when we
    # hit an unimplemented function will get a log like "TODO: implement amdgpu..."
    # instead of a crash when the app tries to call a NULL pointer.
    if len(sys.argv) != 3:
        print('Usage: {} path/to/amdgpu.h header|source'.format(sys.argv[0]))
        sys.exit(-1)

    if sys.argv[2] == 'header':
        generate_header(sys.argv[1])
    elif sys.argv[2] == 'source':
        generate_source(sys.argv[1])
    else:
        print(sys.argv)
        sys.exit(-1)
