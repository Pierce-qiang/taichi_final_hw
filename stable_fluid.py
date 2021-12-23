# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01
# https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py
# https://github.com/erizmr/stable_fluid_MGPCG/blob/master/stable_fluid_MGPCG.py
# https://forum.taichi.graphics/t/homework-0-fluid-simulation/459

import argparse

import numpy as np

import taichi as ti

from PCG_Solver import PCG
# How to run:
#   `python stable_fluid.py`: use the jacobi iteration to solve the linear system.
#   `python stable_fluid.py -S`: use a sparse matrix to do so.
#   `python stable_fluid.py -M`: use MGPCG to do so.
#    press 'A' switch advection reflection
#    press 'C' show curl field
#    press 'V' show velocity field
#    press 'S' enhance vortex
parser = argparse.ArgumentParser()
parser.add_argument('-S',
                    '--use-sp-mat',
                    action='store_true',
                    help='Solve Poisson\'s equation by using a sparse matrix')
parser.add_argument('-M',
                    '--use-MGPCG',
                    action='store_true',
                    help='Solve Poisson\'s equation by using MGPCG')
args, unknowns = parser.parse_known_args()

# params
res = 512
dt = 0.03
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0
debug = False
paused = False
use_sparse_matrix = False
maccormack = True
advection_reflection = False

# TODO: choose mode
use_spray = False    # which change velocity a lot
use_smoke = True   # 2 smoke source
use_impulse = False  # drag impulse


use_sparse_matrix = args.use_sp_mat
# use_sparse_matrix = True
use_pcg = args.use_MGPCG
# use_pcg = True
if use_sparse_matrix:
    ti.init(arch=ti.x64)
    print('Using sparse matrix')
elif use_pcg:
    ti.init(arch=ti.gpu)
    print('Using mgpcg')
else:
    ti.init(arch=ti.gpu)
    print('Using jacobi iteration')
    print(p_jacobi_iters)

_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))

v_aux = ti.Vector.field(2, float, shape=(res, res))

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)




if use_sparse_matrix:
    # use a sparse matrix to solve Poisson's pressure equation.
    @ti.kernel
    def fill_laplacian_matrix(A: ti.linalg.sparse_matrix_builder()):
        for i, j in ti.ndrange(res, res):
            row = i * res + j
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0
                center += 1.0
            if j != res - 1:
                A[row, row + 1] += -1.0
                center += 1.0
            if i != 0:
                A[row, row - res] += -1.0
                center += 1.0
            if i != res - 1:
                A[row, row + res] += -1.0
                center += 1.0
            A[row, row] += center

    N = res * res
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    b = ti.field(ti.f32, shape=N)

    fill_laplacian_matrix(K)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)
#region utility func
@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(res - 1, I))
    return qf[I]
@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)
@ti.func
def sample_max(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return max(a,b,c,d)
@ti.func
def sample_min(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return min(a,b,c,d)
#endregion

# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p_new = p - dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p_new


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), dt:ti.f32):
    for I in ti.grouped(qf):
        p = I + 0.5
        if maccormack:
            q = bilerp(qf, p)  # original q
            p_new = backtrace(vf, p, dt)
            q_new = bilerp(qf, p_new)
            p_aux = backtrace(vf,p_new, -dt)
            q_aux = bilerp(qf, p_aux)
            error = q - q_aux
            q = q_new + 0.5 * error
            # check error rational
            min_val = sample_min(qf, p_new)
            max_val = sample_max(qf, p_new)
            if any(q <min_val) or any(q>max_val):
                q = q_new
            new_qf[I] = q
        else:
            p_new = backtrace(vf, p, dt)
            q_new = bilerp(qf, p_new)
            new_qf[I] = q_new

@ti.kernel
def apply_dye_decay(qf:ti.template()):
    for I in ti.grouped(qf):
        qf[I] = qf[I] * dye_decay

# add impulse and dye
@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  imp_data: ti.ext_arr()):
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j]
        a = dc.norm()

        momentum = mdir * f_strength * factor * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        max_dye = 1
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])
            dc[0] = min(dc[0], max_dye)
            dc[1] = min(dc[1], max_dye)
            dc[2] = min(dc[2], max_dye)
        dyef[i, j] = dc
@ti.kernel
def spray(vf: ti.template(), dyef: ti.template()):
    posx = 256 -10
    posy = 10 -10
    buoyancy = ti.Vector([0.0, 9.8])
    for i ,j in ti.ndrange(20,20, 2):
        dc = dyef[posx+i, posy+j]
        dirx = 2 * ti.random() - 1
        diry = ti.random()
        dir = ti.Vector([dirx, diry])
        momentum = 100 * dir + buoyancy*10
        vf[posx+i, posy+j] = vf[posx+i, posy+j] + momentum
        # add dye
        dc += ti.Vector([0.7, 0.7, 0.7])
        dyef[posx+i, posy+j] = dc
@ti.func
def smooth_step(a, b, x):
    y = (x - a) / (b - a)
    if y < 0.0:
        y = 0.0
    if y > 1.0:
        y = 1.0
    rst = (y * y * (3.0 - 2.0 * y))
    return rst

# https://forum.taichi.graphics/t/homework-0-fluid-simulation/459
@ti.kernel
def add_smoke(x: ti.i32, y: ti.i32, r: ti.i32, value: ti.f32, q:ti.template()):
    for index in range((2 * r + 1) * (2 * r + 1)):
        i = index // (2 * r + 1) - r
        j = ti.mod(index, 2 * r + 1) - r
        q_new = q[x + i, y + j] + value * smooth_step(r * r, 0.0, i * i + j * j)
        q[x + i, y + j] = q_new
# apply buoyancy and decay
@ti.kernel
def apply_buoyancy(vf: ti.template(), df: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        den = df[i, j]
        v[1] += (den.norm() * 25.0 - 5.0) * dt
        # random disturbance
        v[0] += (ti.random(ti.f32) - 0.5) * 80.0
        v[1] += (ti.random(ti.f32) - 0.5) * 80.0
        # velocity damping
        den *= dye_decay
        v *= dye_decay
        vf[i, j] = v
        df[i, j] = den

@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5
        # 0.5 means grid_scale

@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = min(max(vf[i, j] + force * dt, -1e3), 1e3)


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.template()):
    for I in ti.grouped(div_in):
        div_out[I[0] * res + I[1]] = -div_in[I]


@ti.kernel
def apply_pressure(p_in: ti.ext_arr(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res + I[1]]


def solve_pressure_sp_mat():
    copy_divergence(velocity_divs, b)
    x = solver.solve(b)
    apply_pressure(x, pressures_pair.cur)


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

def solve_pressure():
    divergence(velocities_pair.cur)
    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    if use_sparse_matrix:
        solve_pressure_sp_mat()
    elif use_pcg:
        my_solver.init(velocity_divs, -1)
        my_solver.solve(max_iters=2, verbose=False)
        my_solver.fetch_result(pressures_pair.cur)
    else:
        solve_pressure_jacobi()
@ti.kernel
def reflect(advected_v:ti.template(), projected_v:ti.template(), reflected_v:ti.template()):
    for I in ti.grouped(advected_v):
        reflected_v[I] = 2 * projected_v[I] - advected_v[I]
@ti.kernel
def copy_from(src:ti.template(), des:ti.template()):
    for I in ti.grouped(src):
        des[I] = src[I]

def add_dye(mouse_data):
    if use_impulse:
        apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)
        apply_dye_decay(dyes_pair.cur)
        apply_dye_decay(velocities_pair.cur)
    elif use_spray:
        spray(velocities_pair.cur, dyes_pair.cur)
        apply_dye_decay(dyes_pair.cur)
        apply_dye_decay(velocities_pair.cur)
    elif use_smoke:
        add_smoke(int(res* 0.25), 50, 25, 0.8 ,dyes_pair.cur)
        add_smoke(int(res * 0.75), 50, 25, 0.8, dyes_pair.cur)
        apply_buoyancy(velocities_pair.cur, dyes_pair.cur)
    else:
        print("please choose dye mode")
        print("spray, smoke or impulse")
        exit()



def step(mouse_data):
    add_dye(mouse_data)
    if advection_reflection:
        advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt, 0.5 * dt)
        advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, 0.5*dt)
        dyes_pair.swap()
        velocities_pair.swap()

        solve_pressure()
        copy_from(velocities_pair.cur,velocities_pair.nxt)
        subtract_gradient(velocities_pair.cur, pressures_pair.cur)
        reflect(velocities_pair.nxt, velocities_pair.cur,v_aux)
        # now reflected v is v_aux, nxt is advected, cur is projected
        # use div-free vel field as vel, reflect v as q to advect
        advect(velocities_pair.cur, v_aux , velocities_pair.nxt, 0.5*dt)
        advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, 0.5*dt)
        velocities_pair.swap()
        dyes_pair.swap()
        # project again
        solve_pressure()
        subtract_gradient(velocities_pair.cur, pressures_pair.cur)


    else:
        advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt, dt)
        advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, dt)
        dyes_pair.swap()
        velocities_pair.swap()

        solve_pressure()
        subtract_gradient(velocities_pair.cur, pressures_pair.cur)


    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)

visualize_d = True  #visualize dye (default)
visualize_v = False #visualize velocity
visualize_c = False  #visualize curl

gui = ti.GUI('Stable Fluid', (res, res))
md_gen = MouseDataGen()
if use_pcg:
    my_solver = PCG(dim=2,resolution=(res,res),n_mg_levels=3,block_size=16,
                    use_multigrid=True,sparse=False)

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == 'r':
            paused = False
            reset()
        elif e.key == 's':
            if curl_strength:
                curl_strength = 0
            else:
                curl_strength = 7
        elif e.key == 'g':
            gravity = not gravity
        elif e.key == 'v':
            visualize_v = True
            visualize_c = False
            visualize_d = False
        elif e.key == 'd':
            visualize_d = True
            visualize_v = False
            visualize_c = False
        elif e.key == 'c':
            visualize_c = True
            visualize_d = False
            visualize_v = False
        elif e.key == 'p':
            paused = not paused
        elif e.key == 'd':
            debug = not debug
        elif e.key == 'a':
            advection_reflection = not advection_reflection

    # Debug divergence:
    # print(max((abs(velocity_divs.to_numpy().reshape(-1)))))

    if not paused:
        mouse_data = md_gen(gui)
        step(mouse_data)
    if visualize_c:
        vorticity(velocities_pair.cur)
        gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
    elif visualize_d:
        gui.set_image(dyes_pair.cur)
    elif visualize_v:
        gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
    gui.show()