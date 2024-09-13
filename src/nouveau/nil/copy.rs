// Copyright © 2024 Valve Corp.
// SPDX-License-Identifier: MIT

use crate::extent::{units, Extent4D, Offset4D};
use crate::image::Image;
use crate::tiling::{gob_height, Tiling, GOB_DEPTH, GOB_WIDTH_B};
use crate::ILog2Ceil;

pub const SECTOR_WIDTH_B: u32 = 16;
pub const SECTOR_HEIGHT: u32 = 2;
pub const SECTOR_SIZE_B: u32 = SECTOR_WIDTH_B * SECTOR_HEIGHT;

// This file is dedicated to the internal tiling layout, mainly in the context
// of CPU-based tiled memcpy implementations (and helpers) for VK_EXT_host_image_copy
//
// Work here is based on isl_tiled_memcpy, fd6_tiled_memcpy, old work by Rebecca Mckeever,
// and https://fgiesen.wordpress.com/2011/01/17/texture-tiling-and-swizzling/
//
// On NVIDIA, the tiling system is a two-tier one, and images are first tiled in
// a grid of rows of tiles (called "Blocks") with one or more columns:
//
// +----------+----------+----------+----------+
// | Block 0  | Block 1  | Block 2  | Block 3  |
// +----------+----------+----------+----------+
// | Block 4  | Block 5  | Block 6  | Block 7  |
// +----------+----------+----------+----------+
// | Block 8  | Block 9  | Block 10 | Block 11 |
// +----------+----------+----------+----------+
//
// The blocks themselves are ordered linearly as can be seen above, which is
// where the "Block Linear" naming comes from for NVIDIA's tiling scheme.
//
// For 3D images, each block continues in the Z direction such that tiles
// contain multiple Z slices. If the image depth is longer than the
// block depth, there will be more than one layer of blocks, where a layer is
// made up of 1 or more Z slices. For example, if the above tile pattern was
// the first layer of a multilayer arrangement, the second layer would be:
//
// +----------+----------+----------+----------+
// | Block 12 | Block 13 | Block 14 | Block 15 |
// +----------+----------+----------+----------+
// | Block 16 | Block 17 | Block 18 | Block 19 |
// +----------+----------+----------+----------+
// | Block 20 | Block 21 | Block 22 | Block 23 |
// +----------+----------+----------+----------+
//
// The number of rows, columns, and layers of tiles can thus be deduced to be:
//    rows    >= ceiling(image_height / block_height)
//    columns >= ceiling(image_width  / block_width)
//    layers  >= ceiling(image_depth  / block_depth)
//
// Where block_width is a constant 64B (unless for sparse) and block_height
// can be either 8 or 16 GOBs tall (more on GOBs below). For us, block_depth
// is one for now.
//
// The >= is in case the blocks around the edges are partial.
//
// Now comes the second tier. Each block is composed of GOBs (Groups of Bytes)
// arranged in ascending order in a single column:
//
// +---------------------------+
// |           GOB 0           |
// +---------------------------+
// |           GOB 1           |
// +---------------------------+
// |           GOB 2           |
// +---------------------------+
// |           GOB 3           |
// +---------------------------+
//
// The number of GOBs in a full block is
//    block_height * block_depth
//
// An Ampere GOB is 512 bytes, arranged in a 64x8 layout and split into Sectors.
// Each Sector is 32 Bytes, and the Sectors in a GOB are arranged in a 16x2
// layout (i.e., two 16B lines on top of each other). It's then arranged into
// two columns that are 2 sectors by 4, leading to a 4x4 grid of sectors:
//
// +----------+----------+----------+----------+
// | Sector 0 | Sector 1 | Sector 0 | Sector 1 |
// +----------+----------+----------+----------+
// | Sector 2 | Sector 3 | Sector 2 | Sector 3 |
// +----------+----------+----------+----------+
// | Sector 4 | Sector 5 | Sector 4 | Sector 5 |
// +----------+----------+----------+----------+
// | Sector 6 | Sector 7 | Sector 6 | Sector 7 |
// +----------+----------+----------+----------+
//
// From the given pixel address equations in the Orin manual, we arrived at
// the following bit interleave pattern for the pixel address:
//
//      b8 b7 b6 b5 b4 b3 b2 b1 b0
//      --------------------------
//      x5 y2 y1 x4 y0 x3 x2 x1 x0
//
// Which would look something like this:
// fn get_pixel_offset(
//      x: usize,
//      y: usize,
//  ) -> usize {
//      (x & 15)       |
//      (y & 1)  << 4  |
//      (x & 16) << 1  |
//      (y & 2)  << 5  |
//      (x & 32) << 3
//  }
//
//

// The way our implementation will work is by splitting an image into tiles, then
// each tile will be broken into its GOBs, and finally each GOB into sectors,
// where each sector will be copied into its position.
//
// For code sharing and cleanliness, we write everything to be very generic,
// so as to be shared between Linear <-> Tiled and Tiled <-> Linear paths, and
// (ab)use Rust's traits to specialize the last level (copy_gob/copy_whole_gob)
// for a particular direction.
//
// The copy_x and copy_whole_x distinction is made because if we can guarantee
// that tiles/gobs are whole and aligned, we can skip all bounds checking and
// copy things in fast and tight loops

trait LinearTiledCopy {
    // No 3D GOBs for now
    unsafe fn copy_gob(
        x_start: i32,
        y_start: i32,
        x_end: u32,
        y_end: u32,
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    );

    // No bounding box for this one
    unsafe fn copy_whole_gob(
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    );

    unsafe fn copy_tile(
        x_start: i32,
        y_start: i32,
        z_start: i32,
        x_end: u32,
        y_end: u32,
        z_end: u32,
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    ) {
        // Now it is time to break down the tile we have into GOBs. A block
        // is composed of GOBs arranged vertically as follows:
        //      +---------------------------+
        //      |           GOB 0           |
        //      +---------------------------+
        //      |           GOB 1           |
        //      +---------------------------+
        //      |           GOB 2           |
        //      +---------------------------+
        //      |           GOB 3           |
        //      +---------------------------+
        // Normally individual tiles are of tile_width_B = GOB_WIDTH_B, so they
        // already have the correct width, and we just need to divide by
        // gob_height_B, and loop over the vertical dimension. However, in the
        // sparse case, a tile can be multiple GOBs wide, so we also have to
        // account for tile width.

        let tile_extent_B = tiling.extent_B();
        let x_min = std::cmp::max(x_start, 0);
        let x_max = std::cmp::min(x_end, tile_extent_B.width);
        let y_min = std::cmp::max(y_start, 0);
        let y_max = std::cmp::min(y_end, tile_extent_B.height);
        let z_min = std::cmp::max(z_start, 0);
        let z_max = std::cmp::min(z_end, tile_extent_B.depth);

        let gob_extent_B = tiling.gob_extent_B();

        let x_start_gob = x_min / gob_extent_B.width as i32;
        let y_start_gob = y_min / gob_extent_B.height as i32;

        let x_end_gob = x_max.div_ceil(gob_extent_B.width);
        let y_end_gob = y_max.div_ceil(gob_extent_B.height);

        let z_start_tl = z_min / tile_extent_B.depth as i32;
        let z_end_tl = z_max.div_ceil(tile_extent_B.depth);

        for z_tl in z_start_tl..z_end_tl as i32 {
            let z_B = z_tl * tile_extent_B.depth as i32;
            let z_min = z_min - z_B;
            for y_gob in y_start_gob..y_end_gob as i32 {
                let y_B = y_gob as u32 * gob_extent_B.height;
                let y_min = y_min - y_B as i32;
                for x_gob in x_start_gob..x_end_gob as i32 {
                    let x_B = x_gob as u32 * gob_extent_B.width;
                    let x_min = x_min - x_B as i32;
                    let gob_offset: u32 = ((x_B >> 6) & (1 << tiling.x_log2))
                        + ((y_B >> 3) & (1 << tiling.y_log2))
                        + ((z_tl as u32 & (1 << tiling.z_log2))
                            << tiling.y_log2);
                    let tiled = tiled.wrapping_add(
                        (gob_offset
                            * (gob_extent_B.height * gob_extent_B.width))
                            .try_into()
                            .unwrap(),
                    );

                    if x_min <= 0
                        && y_min <= 0
                        && x_max >= gob_extent_B.width.try_into().unwrap()
                        && y_max >= gob_extent_B.height.try_into().unwrap()
                    {
                        Self::copy_whole_gob(
                            tiling,
                            linear_row_stride_B,
                            linear_plane_stride_B,
                            tiled,
                            linear,
                        );
                    } else {
                        Self::copy_gob(
                            x_min,
                            y_min,
                            x_max.try_into().unwrap(),
                            y_max.try_into().unwrap(),
                            tiling,
                            linear_row_stride_B,
                            linear_plane_stride_B,
                            linear,
                            tiled,
                        );
                    }
                }
            }
        }
    }

    // No bounding box for this one
    unsafe fn copy_whole_tile(
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    ) {
        // Now it is time to break down the tile we have into GOBs. A block
        // is composed of GOBs arranged vertically as follows:
        //      +---------------------------+
        //      |           GOB 0           |
        //      +---------------------------+
        //      |           GOB 1           |
        //      +---------------------------+
        //      |           GOB 2           |
        //      +---------------------------+
        //      |           GOB 3           |
        //      +---------------------------+
        // Normally individual tiles are of tile_width_B = GOB_WIDTH_B, so they
        // already have the correct width, and we just need to divide by
        // gob_height_B, and loop over the vertical dimension. However, in the
        // sparse case, a tile can be multiple GOBs wide, so we also have to
        // account for tile width.

        let gob_extent_B = tiling.gob_extent_B();
        let tile_extent_B = tiling.extent_B();
        let tile_extent_GOB = tile_extent_B.to_GOB(tiling.gob_height_is_8);

        for z_tl in 0..tile_extent_GOB.depth as i32 {
            for y_gob in 0..tile_extent_GOB.height {
                let y_B = y_gob * gob_extent_B.height;
                for x_gob in 0..tile_extent_GOB.width {
                    let x_B = x_gob * gob_extent_B.width;
                    let gob_offset: u32 = ((x_B >> 6) & (1 << tiling.x_log2))
                        + ((y_B as u32 >> 3) & (1 << tiling.y_log2))
                        + ((z_tl as u32 & (1 << tiling.z_log2))
                            << tiling.y_log2);

                    let tiled = tiled.wrapping_add(
                        (gob_offset
                            * (gob_extent_B.height * gob_extent_B.width))
                            .try_into()
                            .unwrap(),
                    );

                    Self::copy_whole_gob(
                        tiling,
                        linear_row_stride_B,
                        linear_plane_stride_B,
                        tiled,
                        linear,
                    );
                }
            }
        }
    }

    unsafe fn copy(
        start_px: Offset4D<units::Pixels>,
        extent_px: Extent4D<units::Pixels>,
        miplevel: usize,
        nil: Image,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        Bpp: u8,
        linear: usize,
        tiled: usize,
    ) {
        let start_B = start_px.to_B(nil.format, nil.sample_layout);
        let extent_B = extent_px.to_B(nil.format, nil.sample_layout);

        let tiling: Tiling = nil.levels[miplevel].tiling;
        let tile_size_B = tiling.size_B();
        let tile_extent_B = tiling.extent_B();

        // For the general case, blocks and GOBs are comprised of 9 parts as outlined below:
        //
        //                   x_start   x_whole_tile_start   x_whole_tile_end x_end
        //
        //         y_start    |---------|--------------------------|---------|
        //                    |         |                          |         |
        // y_whole_tile_start |---------|--------------------------|---------|
        //                    |         |                          |         |
        //                    |         |      whole tile area     |         |
        //                    |         |                          |         |
        //   y_whole_tile_end |---------|--------------------------|---------|
        //                    |         |                          |         |
        //      y_end         |---------|--------------------------|---------|
        //
        // The whole block/GOB areas are fully aligned and can be fast-pathed, while
        // the other unaligned/incomplete areas need dedicated handling. So the
        // idea here is to split the image, and use our fast `copy_whole_tile`
        // function for whole tiles, and handle the rest with the normal `copy_tile`

        let x_start = start_B.x as i32;
        let y_start = start_B.y as i32;
        let z_start = start_B.z as i32;

        let x_end = (start_B.x + extent_B.width) as u32;
        let y_end = (start_B.y + extent_B.height) as u32;
        let z_end = (start_B.z + extent_B.depth) as u32;

        let lvl_extent_px = Image::level_extent_px(&nil, miplevel as u32);
        let lvl_extent_tl =
            lvl_extent_px.to_tl(&tiling, nil.format, nil.sample_layout);

        let x_start_tl = x_start / tile_extent_B.width as i32;
        let y_start_tl = y_start / tile_extent_B.height as i32;
        let z_start_tl = z_start / tile_extent_B.depth as i32;

        let x_end_tl = x_end.div_ceil(tile_extent_B.width);
        let y_end_tl = y_end.div_ceil(tile_extent_B.height);
        let z_end_tl = z_end.div_ceil(tile_extent_B.depth);

        for z_tl in z_start_tl..z_end_tl as i32 {
            // These are done here to make the inner loops tighter
            let z_B = z_tl * tile_extent_B.depth as i32;
            let linear = linear.wrapping_add(
                (z_B * linear_plane_stride_B as i32).try_into().unwrap(),
            );
            let tiled = tiled.wrapping_add(
                (z_tl
                    * lvl_extent_tl.width as i32
                    * lvl_extent_tl.height as i32
                    * tile_size_B as i32)
                    .try_into()
                    .unwrap(),
            );
            let z_start = z_start - z_B;
            let z_end = z_end - z_B as u32;
            for y_tl in y_start_tl..y_end_tl as i32 {
                // See above, done here to tighten the inner loop
                let y_B = y_tl * tile_extent_B.height as i32;
                let linear = linear.wrapping_add(
                    (y_B * linear_row_stride_B as i32).try_into().unwrap(),
                );
                let tiled = tiled.wrapping_add(
                    (y_tl * lvl_extent_tl.width as i32 * tile_size_B as i32)
                        .try_into()
                        .unwrap(),
                );
                let y_start = y_start - y_B;
                let y_end = y_end - y_B as u32;
                for x_tl in x_start_tl..x_end_tl as i32 {
                    let x_B = x_tl * tile_extent_B.width as i32;
                    let linear = linear.wrapping_add(x_B.try_into().unwrap());
                    let tiled = tiled.wrapping_add(
                        (x_tl * tile_size_B as i32).try_into().unwrap(),
                    );
                    let x_start = x_start - x_B;
                    let x_end = x_end - x_B as u32;
                    if x_start <= 0
                        && y_start <= 0
                        && z_start <= 0
                        && x_end >= tile_extent_B.width
                        && y_end >= tile_extent_B.height
                        && z_end >= tile_extent_B.depth
                    {
                        Self::copy_whole_tile(
                            tiling,
                            linear_row_stride_B,
                            linear_plane_stride_B,
                            tiled,
                            linear,
                        );
                    } else {
                        Self::copy_tile(
                            x_B,
                            y_B,
                            z_B,
                            x_end,
                            y_end,
                            z_end,
                            tiling,
                            linear_row_stride_B,
                            linear_plane_stride_B,
                            linear,
                            tiled,
                        );
                    }
                }
            }
        }
    }
}
struct CopyTiledToLinear {}

impl LinearTiledCopy for CopyTiledToLinear {
    unsafe fn copy_gob(
        x_start: i32,
        y_start: i32,
        x_end: u32,
        y_end: u32,
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    ) {
        let gob_height = gob_height(tiling.gob_height_is_8);
        let x_start = std::cmp::max(x_start, 0);
        let x_end = std::cmp::min(x_end as u32, GOB_WIDTH_B);
        let y_start = std::cmp::max(y_start, 0);
        let y_end = std::cmp::min(y_end as u32, gob_height);

        let mut x_sector = 0;
        let mut y_sector = 0;

        // A GOB is 512B, and a sector is 32B, so there are 16 sectors in a GOB
        // TODO: Not sure if it's a good idea to have it constant due to generational changes
        for sector_idx in 0..16 {
            let x_min = x_sector - x_start;
            let y_min = y_sector - y_start;

            let tiled = tiled
                .wrapping_add((sector_idx * SECTOR_SIZE_B).try_into().unwrap());
            let linear = linear.wrapping_add(
                (y_min * (GOB_WIDTH_B as i32) + x_min).try_into().unwrap(),
            );

            let x_start = x_start - x_sector;
            let y_start = y_start - y_sector;

            let x_end: u32 = x_end - x_sector as u32;
            let y_end: u32 = y_end - y_sector as u32;

            if x_start <= 0
                && y_start <= 0
                && x_end >= SECTOR_WIDTH_B.try_into().unwrap()
                && y_end >= SECTOR_HEIGHT.try_into().unwrap()
            {
                for y in 0..SECTOR_HEIGHT {
                    unsafe {
                        let src_ptr: usize =
                            tiled + (y * SECTOR_WIDTH_B) as usize;
                        let dst_ptr: usize =
                            linear + (y * GOB_WIDTH_B) as usize;
                        std::ptr::copy_nonoverlapping(
                            src_ptr as *const usize,
                            dst_ptr as *mut usize,
                            SECTOR_WIDTH_B.try_into().unwrap(),
                        );
                    }
                }
            } else {
                let x_min = std::cmp::max(x_start, 0);
                let x_max = std::cmp::min(x_end as u32, SECTOR_WIDTH_B);
                let y_min = std::cmp::max(y_start, 0);
                let y_max = std::cmp::min(y_end as u32, SECTOR_HEIGHT);

                for y in y_start..y_end as i32 {
                    unsafe {
                        let src_ptr: usize = tiled
                            + (y * SECTOR_WIDTH_B as i32 + x_min) as usize;
                        let dst_ptr: usize = linear
                            + ((y - y_min) * GOB_WIDTH_B as i32) as usize;
                        std::ptr::copy_nonoverlapping(
                            src_ptr as *const usize,
                            dst_ptr as *mut usize,
                            x_max.try_into().unwrap(),
                        );
                    }
                }
            }

            // Sectors within a GOB are arranged as follows:
            // +----------+----------+----------+----------+
            // | Sector 0 | Sector 1 | Sector 0 | Sector 1 |
            // +----------+----------+----------+----------+
            // | Sector 2 | Sector 3 | Sector 2 | Sector 3 |
            // +----------+----------+----------+----------+
            // | Sector 4 | Sector 5 | Sector 4 | Sector 5 |
            // +----------+----------+----------+----------+
            // | Sector 6 | Sector 7 | Sector 6 | Sector 7 |
            // +----------+----------+----------+----------+
            // Thus, we need to adhere to this structure as well.

            if sector_idx % 2 == 0 {
                x_sector = x_sector + (SECTOR_WIDTH_B as i32);
            } else if sector_idx % 4 == 1 {
                x_sector = x_sector - (SECTOR_WIDTH_B as i32);
                y_sector = y_sector + (SECTOR_HEIGHT as i32);
            } else if sector_idx == 3 {
                x_sector = x_sector + (SECTOR_WIDTH_B as i32);
                y_sector = y_sector - (SECTOR_HEIGHT as i32);
            }
        }
    }

    // No bounding box for this one
    unsafe fn copy_whole_gob(
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    ) {
        let mut x_sector = 0;
        let mut y_sector = 0;

        // A GOB is 512B, and a sector is 32B, so there are 16 sectors in a GOB
        // TODO: Not sure if it's a good idea to have it constant due to generational changes
        for sector_idx in 0..16 {
            let tiled = tiled
                .wrapping_add((sector_idx * SECTOR_SIZE_B).try_into().unwrap());
            let linear = linear.wrapping_add(
                (y_sector * GOB_WIDTH_B + x_sector).try_into().unwrap(),
            );

            for y in 0..SECTOR_HEIGHT {
                unsafe {
                    let src_ptr: usize = tiled + (y * SECTOR_WIDTH_B) as usize;
                    let dst_ptr: usize = linear + (y * GOB_WIDTH_B) as usize;
                    std::ptr::copy_nonoverlapping(
                        src_ptr as *const usize,
                        dst_ptr as *mut usize,
                        SECTOR_WIDTH_B.try_into().unwrap(),
                    );
                }
            }

            // Sectors within a GOB are arranged as follows:
            // +----------+----------+----------+----------+
            // | Sector 0 | Sector 1 | Sector 0 | Sector 1 |
            // +----------+----------+----------+----------+
            // | Sector 2 | Sector 3 | Sector 2 | Sector 3 |
            // +----------+----------+----------+----------+
            // | Sector 4 | Sector 5 | Sector 4 | Sector 5 |
            // +----------+----------+----------+----------+
            // | Sector 6 | Sector 7 | Sector 6 | Sector 7 |
            // +----------+----------+----------+----------+
            // Thus, we need to adhere to this structure as well.

            if sector_idx % 2 == 0 {
                x_sector = x_sector + SECTOR_WIDTH_B;
            } else if sector_idx % 4 == 1 {
                x_sector = x_sector - SECTOR_WIDTH_B;
                y_sector = y_sector + SECTOR_HEIGHT;
            } else if sector_idx == 3 {
                x_sector = x_sector + SECTOR_WIDTH_B;
                y_sector = y_sector - SECTOR_HEIGHT;
            }
        }
    }
}

struct CopyLinearToTiled {}

impl LinearTiledCopy for CopyLinearToTiled {
    unsafe fn copy_gob(
        x_start: i32,
        y_start: i32,
        x_end: u32,
        y_end: u32,
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    ) {
        let gob_height = gob_height(tiling.gob_height_is_8);
        let x_start = std::cmp::max(x_start, 0);
        let x_end = std::cmp::min(x_end as u32, GOB_WIDTH_B);
        let y_start = std::cmp::max(y_start, 0);
        let y_end = std::cmp::min(y_end as u32, gob_height);

        let mut x_sector = 0;
        let mut y_sector = 0;

        // A GOB is 512B, and a sector is 32B, so there are 16 sectors in a GOB
        // TODO: Not sure if it's a good idea to have it constant due to generational changes
        for sector_idx in 0..16 {
            let x_min = x_sector - x_start;
            let y_min = y_sector - y_start;

            let tiled = tiled
                .wrapping_add((sector_idx * SECTOR_SIZE_B).try_into().unwrap());
            let linear = linear
                .wrapping_add((y_min * (GOB_WIDTH_B as i32) + x_min) as usize);

            let x_start = x_start - x_sector;
            let y_start = y_start - y_sector;

            let x_end: u32 = x_end - x_sector as u32;
            let y_end: u32 = y_end - y_sector as u32;

            if x_start <= 0
                && y_start <= 0
                && x_end >= SECTOR_WIDTH_B.try_into().unwrap()
                && y_end >= SECTOR_HEIGHT.try_into().unwrap()
            {
                for y in 0..SECTOR_HEIGHT {
                    unsafe {
                        let src_ptr: usize =
                            linear + (y * GOB_WIDTH_B) as usize;
                        let dst_ptr: usize =
                            tiled + (y * SECTOR_WIDTH_B) as usize;
                        std::ptr::copy_nonoverlapping(
                            src_ptr as *const usize,
                            dst_ptr as *mut usize,
                            SECTOR_WIDTH_B.try_into().unwrap(),
                        );
                    }
                }
            } else {
                let x_min = std::cmp::max(x_start, 0);
                let x_max = std::cmp::min(x_end as u32, SECTOR_WIDTH_B);
                let y_min = std::cmp::max(y_start, 0);
                let y_max = std::cmp::min(y_end as u32, SECTOR_HEIGHT);

                for y in y_start..y_end as i32 {
                    unsafe {
                        let src_ptr: usize = linear
                            + ((y - y_min) * GOB_WIDTH_B as i32) as usize;
                        let dst_ptr: usize = tiled
                            + (y * SECTOR_WIDTH_B as i32 + x_min) as usize;
                        std::ptr::copy_nonoverlapping(
                            src_ptr as *const usize,
                            dst_ptr as *mut usize,
                            x_max.try_into().unwrap(),
                        );
                    }
                }
            }

            // Sectors within a GOB are arranged as follows:
            // +----------+----------+----------+----------+
            // | Sector 0 | Sector 1 | Sector 0 | Sector 1 |
            // +----------+----------+----------+----------+
            // | Sector 2 | Sector 3 | Sector 2 | Sector 3 |
            // +----------+----------+----------+----------+
            // | Sector 4 | Sector 5 | Sector 4 | Sector 5 |
            // +----------+----------+----------+----------+
            // | Sector 6 | Sector 7 | Sector 6 | Sector 7 |
            // +----------+----------+----------+----------+
            // Thus, we need to adhere to this structure as well.

            if sector_idx % 2 == 0 {
                x_sector = x_sector + (SECTOR_WIDTH_B as i32);
            } else if sector_idx % 4 == 1 {
                x_sector = x_sector - (SECTOR_WIDTH_B as i32);
                y_sector = y_sector + (SECTOR_HEIGHT as i32);
            } else if sector_idx == 3 {
                x_sector = x_sector + (SECTOR_WIDTH_B as i32);
                y_sector = y_sector - (SECTOR_HEIGHT as i32);
            }
        }
    }

    // No bounding box for this one
    unsafe fn copy_whole_gob(
        tiling: Tiling,
        linear_row_stride_B: usize,
        linear_plane_stride_B: usize,
        linear: usize,
        tiled: usize,
    ) {
        let mut x_sector = 0;
        let mut y_sector = 0;

        // A GOB is 512B, and a sector is 32B, so there are 16 sectors in a GOB
        // TODO: Not sure if it's a good idea to have it constant due to generational changes
        for sector_idx in 0..16 {
            let tiled = tiled
                .wrapping_add((sector_idx * SECTOR_SIZE_B).try_into().unwrap());
            let linear = linear.wrapping_add(
                (y_sector * GOB_WIDTH_B + x_sector).try_into().unwrap(),
            );

            for y in 0..SECTOR_HEIGHT {
                unsafe {
                    let src_ptr: usize = linear + (y * GOB_WIDTH_B) as usize;
                    let dst_ptr: usize = tiled + (y * SECTOR_WIDTH_B) as usize;
                    std::ptr::copy_nonoverlapping(
                        src_ptr as *const usize,
                        dst_ptr as *mut usize,
                        SECTOR_WIDTH_B.try_into().unwrap(),
                    );
                }
            }

            // Sectors within a GOB are arranged as follows:
            // +----------+----------+----------+----------+
            // | Sector 0 | Sector 1 | Sector 0 | Sector 1 |
            // +----------+----------+----------+----------+
            // | Sector 2 | Sector 3 | Sector 2 | Sector 3 |
            // +----------+----------+----------+----------+
            // | Sector 4 | Sector 5 | Sector 4 | Sector 5 |
            // +----------+----------+----------+----------+
            // | Sector 6 | Sector 7 | Sector 6 | Sector 7 |
            // +----------+----------+----------+----------+
            // Thus, we need to adhere to this structure as well.

            if sector_idx % 2 == 0 {
                x_sector = x_sector + SECTOR_WIDTH_B;
            } else if sector_idx % 4 == 1 {
                x_sector = x_sector - SECTOR_WIDTH_B;
                y_sector = y_sector + SECTOR_HEIGHT;
            } else if sector_idx == 3 {
                x_sector = x_sector + SECTOR_WIDTH_B;
                y_sector = y_sector - SECTOR_HEIGHT;
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn nil_copy_linear_to_tiled(
    start_px: Offset4D<units::Pixels>,
    extent_px: Extent4D<units::Pixels>,
    miplevel: usize,
    nil: Image,
    linear_row_stride_B: usize,
    linear_plane_stride_B: usize,
    Bpp: u8,
    linear: usize,
    tiled: usize,
) {
    let copy = CopyLinearToTiled {};
    CopyLinearToTiled::copy(
        start_px,
        extent_px,
        miplevel,
        nil,
        linear_row_stride_B,
        linear_plane_stride_B,
        Bpp,
        linear,
        tiled,
    );
}

#[no_mangle]
pub unsafe extern "C" fn nil_copy_tiled_to_linear(
    start_px: Offset4D<units::Pixels>,
    extent_px: Extent4D<units::Pixels>,
    miplevel: usize,
    nil: Image,
    linear_row_stride_B: usize,
    linear_plane_stride_B: usize,
    Bpp: u8,
    linear: usize,
    tiled: usize,
) {
    let copy = CopyTiledToLinear {};
    CopyTiledToLinear::copy(
        start_px,
        extent_px,
        miplevel,
        nil,
        linear_row_stride_B,
        linear_plane_stride_B,
        Bpp,
        linear,
        tiled,
    );
}
