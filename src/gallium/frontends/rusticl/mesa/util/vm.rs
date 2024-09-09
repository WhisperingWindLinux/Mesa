use std::{
    num::NonZeroU64,
    ops::{Deref, DerefMut},
    pin::Pin,
    sync::Mutex,
};

use mesa_rust_gen::*;

pub struct VMInner {
    vm: Pin<Box<util_vma_heap>>,
}

// SAFETY: util_vma_heap is safe to be send between threads.
unsafe impl Send for VMInner {}

impl VMInner {
    pub fn alloc(&mut self, size: u64, alignment: u64) -> Option<VMA> {
        let addr = unsafe { util_vma_heap_alloc(self.vm.deref_mut(), size, alignment) };
        NonZeroU64::new(addr).map(|addr| VMA { vma: addr })
    }

    // TODO: to guarantee a safe interface we should rather return a new object from alloc owning
    // a reference to the vm and take care of the free via drop.
    pub fn free(&mut self, address: u64, size: u64) {
        unsafe {
            util_vma_heap_free(self.vm.deref_mut(), address, size);
        }
    }

    fn new(start: u64, size: u64) -> Self {
        let mut vm = Box::pin(util_vma_heap::default());

        // Safety: util_vma_heap is a linked list with itself (or rather one of its members) as the
        //         start/end, therefore we have to pin the allocation so that it's address
        //         changes.
        unsafe {
            util_vma_heap_init(vm.deref_mut(), start, size);
        }

        Self { vm: vm }
    }
}

impl Drop for VMInner {
    fn drop(&mut self) {
        unsafe {
            util_vma_heap_finish(self.vm.deref_mut());
        }
    }
}

// TODO: make this not suck so much
//       the rough idea on what needs to change here is that VMA takes a reference to the inner
//       mutex and uses it in drop to remove itself from the heap.
pub struct VM {
    // We need to pin the vma_heap because it's part of a linked list and cannot change its
    // location.
    vm: Mutex<VMInner>,
}

impl Deref for VM {
    type Target = Mutex<VMInner>;

    fn deref(&self) -> &Self::Target {
        &self.vm
    }
}

impl VM {
    pub fn new(start: u64, size: u64) -> Self {
        VM {
            vm: Mutex::new(VMInner::new(start, size)),
        }
    }
}

pub struct VMA {
    vma: NonZeroU64,
}

impl VMA {
    pub fn address(&self) -> NonZeroU64 {
        self.vma
    }
}
