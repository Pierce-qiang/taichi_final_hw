import taichi as ti
# Reference
# https://github.com/erizmr/stable_fluid_MGPCG/blob/master/stable_fluid_MGPCG.py

# this solver is for pressure i.e Ap = -d
# Notice: normally bottom_smoothing = 50 ,here just use 10
# for higher precision, increase iterator
# usage: init, solve, fetch_result
@ti.data_oriented
class PCG:
    def __init__(self,
                 dim=2,
                 resolution=(512, 512),
                 n_mg_levels=5,
                 offset=None,
                 block_size=8,
                 use_multigrid=True,
                 sparse=False):
        # params and control
        self.use_multigrid = use_multigrid
        assert len(resolution) == dim
        self.res = resolution
        self.N_multigrid = []
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.dim = dim
        self.real = ti.f32
        self.sparse = sparse

        if offset is None:
            self.offset = [-n // 2 for n in self.res]
        else:
            self.offset = offset
            assert len(offset) == self.dim

        # field
        self.r = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]   # residual
        self.z = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]   # M^-1 r auxiliary vec

        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)     # matrix-vector product
        self.alpha = ti.field(dtype=self.real, shape=())     # step size
        self.beta = ti.field(dtype=self.real, shape=())  # step size
        self.sum = ti.field(dtype=self.real, shape=())  # storage for reductions
        self.old_zTr = ti.field(dtype=self.real, shape=())
        self.new_zTr = ti.field(dtype=self.real, shape=())

        indices = ti.ijk if self.dim == 3 else ti.ij

        coarsened_offset = self.offset
        coarsened_grid_size = list(self.res)
        self.grids = []

        for l in range(self.n_mg_levels):
            self.N_multigrid.append(coarsened_grid_size)
            sparse_grid_size = [
                dim_size + block_size * 2 for dim_size in coarsened_grid_size
            ]
            sparse_grid_offset = [o - block_size for o in coarsened_offset]
            print(f'Level {l}')
            print(f'  coarsened_grid_size {coarsened_grid_size}')
            print(f'  coarsened_offset {coarsened_offset}')

            grid = None
            if sparse:
                grid = ti.root.pointer(
                    indices,
                    [dim_size // block_size for dim_size in sparse_grid_size])
            else:
                grid = ti.root.dense(indices, [
                    dim_size // block_size for dim_size in coarsened_grid_size
                ])

            fields = []

            if l == 0:
                # Finest grid
                fields += [self.x, self.p, self.Ap]
            fields += [self.r[l], self.z[l]]

            for f in fields:
                grid.dense(indices, block_size).place(f)

            self.grids.append(grid)

            new_coarsened_offset = []
            for o in coarsened_offset:
                new_coarsened_offset.append(o // 2)
            coarsened_offset = new_coarsened_offset

            new_coarsened_grid_size = []
            for d in coarsened_grid_size:
                new_coarsened_grid_size.append(d // 2)
            coarsened_grid_size = new_coarsened_grid_size

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    # just pass velocity diversity and * -1
    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        for I in ti.grouped(r):
            self.init_r(I, r[I] * k)

    # return result
    @ti.kernel
    def fetch_result(self, x: ti.template()):
        for I in ti.grouped(x):
            x[I] = self.x[I]

    @ti.func
    def sample(self, x, I):
        res = ti.Vector(x.shape)
        # Add Neumann boundary condition
        II = ti.max(0, ti.min(res - 1, I))
        for D in ti.static(range(self.dim)):
            II[D] = ti.assume_in_range(II[D], I[D], -1, 1)
        return x[II]

    @ti.func
    def neighbor_sum(self, x, I):
        ret = ti.cast(0.0, self.real)
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += self.sample(x, I + offset) + self.sample(x, I - offset)
        return ret

    @ti.kernel
    def compute_Ap(self):
        # Enable block local for sparse allocation
        if ti.static(self.sparse):
            ti.block_local(self.p)  # Hints Taichi to cache the fields and to enable the BLS optimization.
        for I in ti.grouped(self.Ap):
            self.Ap[I] = 2 * self.dim * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        ti.block_dim(32)
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    # @ti.kernel
    # def pre_multiply(self, p: ti.template(), q: ti.template()):
    #     for I in ti.grouped(p):
    #         self.pre_multiply_cache[I] = p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def compute_alpha(self, eps: ti.template()):
        self.sum[None] = 0.0
        ti.block_dim(32)
        for I in ti.grouped(self.p):
            self.sum[None] += self.p[I] * self.Ap[I]
        self.alpha[None] = self.old_zTr[None] / max(self.sum[None], eps)

    @ti.kernel
    def compute_rTr(self, iter: ti.i32, verbose: ti.template()) -> ti.f32:
        rTr = 0.0
        ti.block_dim(32)
        for I in ti.grouped(self.r[0]):
            rTr += self.r[0][I] * self.r[0][I]
        if verbose:
            print('iter', iter, '|residual|_2=', ti.sqrt(rTr))
        return rTr

    @ti.kernel
    def compute_beta(self, eps: ti.template()):
        # beta = new_rTr / old_rTr
        self.new_zTr[None] = 0
        ti.block_dim(32)
        for I in ti.grouped(self.r[0]):
            self.new_zTr[None] += self.z[0][I] * self.r[0][I]
        self.beta[None] = self.new_zTr[None] / max(self.old_zTr[None], eps)

    @ti.kernel
    def update_zTr(self):
        self.old_zTr[None] = self.new_zTr[None]

    @ti.kernel
    def restrict(self, l: ti.template()):
        # Enable block local for sparse allocation
        if ti.static(self.sparse):
            ti.block_local(self.z[l])
        for I in ti.grouped(self.r[l]):
            # r - Ax
            residual = self.r[l][I] - (2 * self.dim * self.z[l][I] -
                                       self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += residual * 0.5

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] = self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red-black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                # z is M-1*r, actually the solution of Ax = r
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / (2 * self.dim)

    def apply_preconditioner(self):
        # multi grid do not use the initial guess
        # remember set zero
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self,
              max_iters=-1,
              eps=1e-12,
              abs_tol=1e-12,
              rel_tol=1e-12,
              iter_batch_size=2,
              verbose=False):

        self.reduce(self.r[0], self.r[0])
        residuals = []
        initial_rTr = self.sum[None]
        residuals.append(initial_rTr)

        tol = max(abs_tol, initial_rTr * rel_tol)

        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()
        self.compute_beta(eps)
        self.update_zTr()

        # Conjugate gradients
        iter = 0
        while max_iters == -1 or iter < max_iters:
            self.compute_Ap()
            self.compute_alpha(eps)

            # x += alpha p
            self.update_x()

            # r -= -alpha A p
            self.update_r()

            # ti.async_flush()
            if iter % iter_batch_size == iter_batch_size - 1:
                rTr = self.compute_rTr(iter, verbose)
                residuals.append(rTr)
                if rTr < tol:
                    break

            # z = M^-1 r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            self.compute_beta(eps)
            # p = z + beta p
            self.update_p()
            self.update_zTr()

            iter += 1
        return residuals