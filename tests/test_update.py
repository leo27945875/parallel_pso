import cuPSO
import unittest
import numpy as np


class TestUpdateFuncs(unittest.TestCase):
    def setUp(self):
        print(f"\nTesting: [{__class__.__name__}] {self._testMethodName}  ", end="")

    def test_calc_fitness_vals_npy(self):
        n, d = 203, 83
        xs_cpu = np.ones((n, d))
        out_cpu = np.zeros(n)
        cuPSO.calc_fitness_vals_npy(xs_cpu, out_cpu)
        self.assertTrue(np.allclose(out_cpu, 0.))

    def test_calc_fitness_vals(self):
        n, d = 203, 83
        xs = cuPSO.Buffer(n, d)
        out = cuPSO.Buffer(n)
        xs.fill(1.)
        out.fill(0.)
        cuPSO.calc_fitness_vals(xs, out)
        out.copy_to_numpy(out_cpu := np.zeros(n))
        self.assertTrue(np.allclose(out_cpu, 0.))

    def test_update_velocities(self):
        n, d, v_max = 203, 83, 10.
        xs = cuPSO.Buffer(n, d)
        vs = cuPSO.Buffer(n, d)
        local_best_xs = cuPSO.Buffer(n, d)
        global_best_x = cuPSO.Buffer(1, d)
        v_sum_pow2 = cuPSO.Buffer(n)
        rng_states = cuPSO.CURANDStates(n * d, 0)

        vs_cpu = np.zeros((n, d))
        v_sum_pow2_cpu = np.zeros(n)

        xs.fill(1.)
        vs.fill(3.5)
        local_best_xs.fill(1.)
        global_best_x.fill(1.)
        v_sum_pow2.fill(0.)

        vs.copy_to_numpy(vs_cpu)
        true_norm = np.linalg.norm(vs_cpu, axis=1)

        cuPSO.update_velocities(xs, vs, local_best_xs, global_best_x, v_sum_pow2, 1., 1., 1., v_max, rng_states)
        vs.copy_to_numpy(vs_cpu)
        v_sum_pow2.copy_to_numpy(v_sum_pow2_cpu)

        self.assertTrue(np.allclose(np.linalg.norm(vs_cpu, axis=1), v_max))
        self.assertTrue(np.allclose(v_sum_pow2_cpu**0.5, true_norm))

    def test_update_positions(self):
        n, d, x_min, x_max = 203, 83, -10., 10.
        xs = cuPSO.Buffer(n, d)
        vs = cuPSO.Buffer(n, d)

        xs.fill(3.)
        vs.fill(-15.)
        cuPSO.update_positions(xs, vs, x_min, x_max)
        xs.copy_to_numpy(xs_cpu := np.zeros((n, d)))
        self.assertTrue(np.allclose(xs_cpu, -10.))

        xs.fill(-3.)
        vs.fill(15.)
        cuPSO.update_positions(xs, vs, x_min, x_max)
        xs.copy_to_numpy(xs_cpu := np.zeros((n, d)))
        self.assertTrue(np.allclose(xs_cpu, 10.))
    
    def test_update_bests(self):
        n, d = 203, 83
        xs = cuPSO.Buffer(n, d)
        local_best_xs = cuPSO.Buffer(n, d)
        global_best_x = cuPSO.Buffer(1, d)
        x_fits = cuPSO.Buffer(n)
        local_best_fits = cuPSO.Buffer(n)
        global_best_fit = cuPSO.Buffer(1)

        xs_cpu = np.arange(n, dtype=np.float64).reshape(-1, 1).repeat(d, axis=1)
        local_best_xs_cpu = np.zeros((n, d))
        global_best_x_cpu = np.ones(d) * -999.
        x_fits_cpu = np.abs(np.arange(n, dtype=np.float64) - n * 0.5)
        local_best_fits_cpu = np.zeros(n) + n * 0.25
        global_best_fit_cpu = np.ones(1) * float("inf")

        true_global_best_idx = (n + 1) // 2
        true_global_best_x = xs_cpu[true_global_best_idx]
        true_global_best_fit = x_fits_cpu[true_global_best_idx]
        true_local_best_xs = np.where((x_fits_cpu <= local_best_fits_cpu)[:, None], xs_cpu, local_best_xs_cpu)
        true_local_best_fits = np.where((x_fits_cpu <= local_best_fits_cpu), x_fits_cpu, local_best_fits_cpu)

        xs.copy_from_numpy(xs_cpu)
        local_best_xs.copy_from_numpy(local_best_xs_cpu)
        global_best_x.copy_from_numpy(global_best_x_cpu)
        x_fits.copy_from_numpy(x_fits_cpu)
        local_best_fits.copy_from_numpy(local_best_fits_cpu)
        global_best_fit.copy_from_numpy(global_best_fit_cpu)

        global_best_idx = cuPSO.update_bests(
            xs, x_fits, local_best_xs, local_best_fits, global_best_x, global_best_fit
        )

        xs.copy_to_numpy(xs_cpu)
        local_best_xs.copy_to_numpy(local_best_xs_cpu)
        global_best_x.copy_to_numpy(global_best_x_cpu)
        x_fits.copy_to_numpy(x_fits_cpu)
        local_best_fits.copy_to_numpy(local_best_fits_cpu)
        global_best_fit.copy_to_numpy(global_best_fit_cpu)

        self.assertTrue(global_best_idx == true_global_best_idx)
        self.assertTrue(np.allclose(global_best_x_cpu, true_global_best_x))
        self.assertTrue(np.allclose(global_best_fit_cpu, true_global_best_fit))
        self.assertTrue(np.allclose(local_best_xs_cpu, true_local_best_xs))
        self.assertTrue(np.allclose(local_best_fits_cpu, true_local_best_fits))