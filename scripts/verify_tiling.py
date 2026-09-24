#!/usr/bin/env python3
"""Verify the tiling, guard and alignment logic of src/gemm.cu without a GPU.

The CUDA-specific parts (cp.async, barriers) cannot be checked here, but the
index arithmetic is where tiling bugs live. This script mirrors the arithmetic
of gemm_tail_kernel / gemm_fast_kernel and checks four things:

  1. every output element is written exactly once by exactly one thread
  2. the guarded shared-memory tile equals the zero-padded A/B block
  3. the 8x8 register accumulation reproduces alpha * A @ B + beta * C
  4. every 16-byte global and shared access is 16-byte aligned

Usage: python3 scripts/verify_tiling.py
"""

BK = 8
APAD = 4
WARP = 32
MIN_TILES = 24

TILE_CONFIGS = {
    "128x128": (128, 128),
    "128x64": (128, 64),
    "64x128": (64, 128),
    "64x64": (64, 64),
}

failures = []


def check(ok, label):
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        failures.append(label)


def threads_for(bm, bn):
    return (bm // 64) * (bn // 32) * WARP


def thread_map(bm, bn):
    """(thread_row, thread_col) for every tid, mirroring the kernel."""
    warps_n = bn // 32
    out = []
    for tid in range(threads_for(bm, bn)):
        warp = tid // WARP
        lane = tid % WARP
        warp_m = warp // warps_n
        warp_n = warp % warps_n
        lane_m = lane // 4
        lane_n = lane % 4
        out.append((warp_m * 64 + lane_m, warp_n * 32 + lane_n * 8))
    return out


def grid_for(bm, bn, m, n):
    return (n + bn - 1) // bn, (m + bm - 1) // bm


def coverage(bm, bn, m, n):
    """Map every (row, col) the tail epilogue writes to the owning threads."""
    owners = {}
    grid_x, grid_y = grid_for(bm, bn, m, n)
    for by in range(grid_y):
        for bx in range(grid_x):
            block_row, block_col = by * bm, bx * bn
            rows_left, cols_left = m - block_row, n - block_col
            for tid, (tr, tc) in enumerate(thread_map(bm, bn)):
                for i in range(8):
                    local_row = tr + i * 8
                    if local_row >= rows_left:
                        break
                    limit = min(8, cols_left - tc)
                    for j in range(limit):
                        key = (block_row + local_row, block_col + tc + j)
                        owners.setdefault(key, []).append(tid)
    return owners


def fill_tile(bm, bn, vec, a, b, tile_k, block_row, block_col, smem_a, smem_b, m, n, k):
    """Mirror of the guarded global -> shared copy, including the zero fill.

    The two-stage rotation of the real kernel only decides *when* a tile is
    loaded; the contents of the buffer a tile is read from do not depend on the
    stage, so the simulation materialises one buffer per K tile instead.
    """
    for row in range(bm):
        for col in range(0, BK, 4):
            grow, gcol = block_row + row, tile_k * BK + col
            if vec:
                valid = grow < m and gcol + 4 <= k
                for t in range(4):
                    smem_a[row][col + t] = a[grow][gcol + t] if valid else 0.0
            else:
                for t in range(4):
                    g = gcol + t
                    smem_a[row][col + t] = a[grow][g] if (grow < m and g < k) else 0.0
    for row in range(BK):
        for col in range(0, bn, 4):
            grow, gcol = tile_k * BK + row, block_col + col
            if vec:
                valid = grow < k and gcol + 4 <= n
                for t in range(4):
                    smem_b[row][col + t] = b[grow][gcol + t] if valid else 0.0
            else:
                for t in range(4):
                    g = gcol + t
                    smem_b[row][col + t] = b[grow][g] if (grow < k and g < n) else 0.0


def simulate_tail(bm, bn, vec, a, b, c0, alpha, beta):
    """Full numeric replay of gemm_tail_kernel for one tile configuration."""
    m, k, n = len(a), len(a[0]) if a else 0, len(c0[0])
    out = [row[:] for row in c0]
    tiles = (k + BK - 1) // BK
    grid_x, grid_y = grid_for(bm, bn, m, n)

    for by in range(grid_y):
        for bx in range(grid_x):
            block_row, block_col = by * bm, bx * bn
            rows_left, cols_left = m - block_row, n - block_col

            buffers = []
            for tile in range(tiles):
                smem_a = [[0.0] * (BK + APAD) for _ in range(bm)]
                smem_b = [[0.0] * bn for _ in range(BK)]
                fill_tile(bm, bn, vec, a, b, tile, block_row, block_col, smem_a, smem_b,
                          m, n, k)
                buffers.append((smem_a, smem_b))

            for tr, tc in thread_map(bm, bn):
                accum = [[0.0] * 8 for _ in range(8)]
                for smem_a, smem_b in buffers:
                    for kk in range(0, BK, 4):
                        ar = [[smem_a[tr + i * 8][kk + k2] for k2 in range(4)]
                              for i in range(8)]
                        for k2 in range(4):
                            brv = [smem_b[kk + k2][tc + j] for j in range(8)]
                            for i in range(8):
                                row = accum[i]
                                av = ar[i][k2]
                                for j in range(8):
                                    row[j] += av * brv[j]
                for i in range(8):
                    local_row = tr + i * 8
                    if local_row >= rows_left:
                        break
                    limit = min(8, cols_left - tc)
                    for j in range(limit):
                        r, c = block_row + local_row, block_col + tc + j
                        out[r][c] = alpha * accum[i][j] + beta * out[r][c]
    return out


def reference(a, b, c0, alpha, beta):
    m, k, n = len(a), len(a[0]) if a else 0, len(c0[0])
    return [[alpha * sum(a[i][t] * b[t][j] for t in range(k)) + beta * c0[i][j]
             for j in range(n)] for i in range(m)]


def make_matrix(rows, cols, seed):
    return [[(((seed * 7919 + i * 131 + j * 17) % 200) - 100) / 100.0
             for j in range(cols)] for i in range(rows)]


def align_problems(bm, bn, m, n, k, vec):
    problems = []
    for row in range(bm):
        for col in range(0, BK, 4):
            if (row * (BK + APAD) + col) * 4 % 16:
                problems.append(("smem_a", row, col))
    for row in range(BK):
        for col in range(0, bn, 4):
            if (row * bn + col) * 4 % 16:
                problems.append(("smem_b", row, col))
    if vec:
        for row in range(bm):
            for col in range(0, BK, 4):
                if (row * k + col) % 4:
                    problems.append(("gmem_a", row, col))
        for row in range(BK):
            for col in range(0, bn, 4):
                if (row * n + col) % 4:
                    problems.append(("gmem_b", row, col))
        for bx in range((n + bn - 1) // bn):
            for tr, tc in thread_map(bm, bn):
                for i in range(8):
                    if (tr + i * 8) * n % 4 or (bx * bn + tc) % 4:
                        problems.append(("gmem_c", tr, tc, i))
    return problems


def dispatch_kind(m, n, k):
    """Mirror of fp32_gemm::launch, returning (path, bm, bn)."""
    if k > 0 and k % BK == 0:
        if m >= 2 * n and m % 128 == 0 and n % 64 == 0:
            return ("fast", 128, 64)
        if n >= 2 * m and m % 64 == 0 and n % 128 == 0:
            return ("fast", 64, 128)
        if m % 128 == 0 and n % 128 == 0:
            return ("fast", 128, 128)
    tiles_128 = ((m + 127) // 128) * ((n + 127) // 128)
    if m >= 128 and n >= 128 and tiles_128 >= MIN_TILES:
        return ("tail", 128, 128)
    if m >= 64 and n >= 64:
        tiles_64 = ((m + 63) // 64) * ((n + 63) // 64)
        if tiles_64 >= MIN_TILES // 2:
            if m >= 128:
                return ("tail", 128, 64)
            if n >= 128:
                return ("tail", 64, 128)
            return ("tail", 64, 64)
    return ("edge", 16, 16)


def main():
    print("== 1. epilogue coverage: every output written exactly once ==")
    cases = [(bm, bn, m, n)
             for bm, bn in TILE_CONFIGS.values()
             for m, n in [(1, 1), (63, 65), (64, 64), (65, 129), (127, 255),
                          (128, 128), (129, 200), (200, 200), (257, 193)]]
    bad = []
    for bm, bn, m, n in cases:
        owners = coverage(bm, bn, m, n)
        if len(owners) != m * n:
            bad.append((bm, bn, m, n, "missing", m * n - len(owners)))
            continue
        dup = [(key, len(v)) for key, v in owners.items() if len(v) != 1]
        if dup:
            bad.append((bm, bn, m, n, "duplicate", dup[0]))
    check(not bad, "%d tile/shape combinations covered exactly once" % len(cases))
    for entry in bad[:5]:
        print("        %s" % (entry,))

    print("== 2. alignment of every 16-byte access ==")
    bad = []
    for bm, bn, m, n in cases:
        for k in (8, 16, 21, 24):
            vec = (k % 4 == 0 and n % 4 == 0)
            problems = align_problems(bm, bn, m, n, k, vec)
            if problems:
                bad.append((bm, bn, m, n, k, problems[0]))
    check(not bad, "all vector accesses are 16-byte aligned")
    for entry in bad[:5]:
        print("        %s" % (entry,))

    print("== 3. numeric replay of gemm_tail_kernel ==")
    numeric = [
        (64, 64, 70, 68, 21),
        (64, 64, 65, 129, 8),
        (64, 64, 64, 64, 0),
        (128, 64, 130, 70, 24),
        (64, 128, 70, 130, 16),
        (128, 128, 129, 200, 24),
        (128, 128, 128, 128, 8),
        (128, 128, 129, 129, 9),
    ]
    for index, (bm, bn, m, n, k) in enumerate(numeric):
        vec = k % 4 == 0 and n % 4 == 0
        a = make_matrix(m, k, index + 1)
        b = make_matrix(k, n, index + 11)
        c0 = make_matrix(m, n, index + 21)
        alpha, beta = 1.25, -0.5
        got = simulate_tail(bm, bn, vec, a, b, c0, alpha, beta)
        want = reference(a, b, c0, alpha, beta)
        worst, where = 0.0, None
        for i in range(m):
            for j in range(n):
                err = abs(got[i][j] - want[i][j])
                if err > worst:
                    worst, where = err, (i, j, want[i][j], got[i][j])
        label = "tail %dx%d vec=%d m=%d n=%d k=%d (max err %.2e)" % (
            bm, bn, vec, m, n, k, worst)
        check(worst < 1e-4, label)
        if worst >= 1e-4 and where:
            print("        first mismatch at %s" % (where,))

    print("== 4. dispatch decisions ==")
    shapes = [(128, 128, 128), (256, 256, 256), (384, 384, 384), (512, 512, 512),
              (768, 768, 768), (1024, 1024, 1024), (1536, 1536, 1536),
              (2048, 2048, 2048), (3072, 3072, 3072), (4096, 4096, 4096),
              (128, 4096, 4096), (256, 4096, 4096), (512, 4096, 4096),
              (1024, 4096, 4096), (4096, 128, 4096), (4096, 256, 4096),
              (4096, 512, 4096), (4096, 1024, 4096), (256, 11008, 4096),
              (512, 11008, 4096), (1024, 11008, 4096), (2048, 11008, 4096),
              (4096, 11008, 4096), (256, 4096, 11008), (512, 4096, 11008),
              (1024, 4096, 11008), (2048, 4096, 11008), (4096, 4096, 11008),
              (1024, 3072, 4096), (2048, 3072, 4096), (3072, 1024, 4096),
              (3072, 2048, 4096), (4096, 3072, 11008),
              (127, 255, 63), (511, 769, 257), (1000, 1000, 1000),
              (1537, 1025, 769), (33, 65, 17)]
    tail_cases = [(511, 769, 257), (1000, 1000, 1000), (1537, 1025, 769)]
    for m, n, k in shapes:
        path, bm, bn = dispatch_kind(m, n, k)
        label = "%-18s -> %-4s %dx%d" % ("%dx%dx%d" % (m, n, k), path, bm, bn)
        if (m, n, k) in tail_cases:
            check(path == "tail", label)
        else:
            print("        " + label)
    check(dispatch_kind(127, 255, 63)[0] == "edge", "127x255x63 keeps the compact edge kernel")
    check(dispatch_kind(33, 65, 17)[0] == "edge", "33x65x17 keeps the compact edge kernel")
    check(dispatch_kind(4096, 4096, 1000)[0] == "fast",
          "4096x4096x1000 (K % 8 == 0) still takes the fast kernel")
    check(dispatch_kind(4096, 4096, 1001)[0] == "tail",
          "4096x4096x1001 (K % 8 != 0) now avoids the edge kernel")

    print()
    if failures:
        print("FAILED: %d check(s)" % len(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
