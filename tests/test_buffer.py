import cuPSO
import unittest
import numpy as np


class TestBuffer(unittest.TestCase):
    def setUp(self):
        print(f"\nTesting: [{__class__.__name__}] {self._testMethodName}  ", end="")

    def test_buffer_size(self):
        buffer = cuPSO.Buffer(200, 50, cuPSO.Device.CPU)
        self.assertEqual(buffer.buffer_size(), 200 * 50 * 8)
        self.assertEqual(buffer.num_elem(), 200 * 50)
        self.assertEqual(buffer.shape(), (200, 50))
        self.assertEqual(buffer.nrow(), 200)
        self.assertEqual(buffer.ncol(), 50)
        self.assertEqual(buffer.device(), cuPSO.Device.CPU)
    
    def test_device_definitions(self):
        self.assertEqual(cuPSO.Device.CPU.name, "CPU")
        self.assertEqual(cuPSO.Device.GPU.name, "GPU")
        self.assertEqual(cuPSO.Device.CPU.value, 0)
        self.assertEqual(cuPSO.Device.GPU.value, 1)

    def test_cpu_buffer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.fill(5.)
        self.assertEqual(buffer[0, 0], 5.)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)
    
    def test_gpu_buffer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.GPU)
        buffer.fill(5.)
        self.assertEqual(buffer[0, 0], 5.)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)

    def test_buffer_transfer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.fill(5.)
        buffer.to(cuPSO.Device.CPU)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)
        buffer.to(cuPSO.Device.GPU)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)
        buffer.to(cuPSO.Device.GPU)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)
        buffer.to(cuPSO.Device.CPU)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)

    def test_cpu_to_numpy(self):
        npy_buffer = np.zeros(10 * 15)
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.fill(5.)
        buffer.copy_to_numpy(npy_buffer)
        self.assertTrue(np.all(npy_buffer == 5.))
    
    def test_gpu_to_numpy(self):
        npy_buffer = np.zeros(10 * 15)
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.GPU)
        buffer.fill(5.)
        buffer.copy_to_numpy(npy_buffer)
        self.assertTrue(np.all(npy_buffer == 5.))

    def test_cpu_from_numpy(self):
        i_npy_buffer, o_npy_buffer = np.zeros(10 * 15) - 10., np.zeros(10 * 15)
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.copy_from_numpy(i_npy_buffer)
        buffer.copy_to_numpy(o_npy_buffer)
        self.assertTrue(np.all(o_npy_buffer == -10.))
    
    def test_gpu_from_numpy(self):
        i_npy_buffer, o_npy_buffer = np.zeros(10 * 15) - 10., np.zeros(10 * 15)
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.GPU)
        buffer.copy_from_numpy(i_npy_buffer)
        buffer.copy_to_numpy(o_npy_buffer)
        self.assertTrue(np.all(o_npy_buffer == -10.))

    def test_clear_cpu_buffer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.clear()
        self.assertEqual(buffer.buffer_size(), 0)
        self.assertEqual(buffer.num_elem(), 0)
        self.assertEqual(buffer.shape(), (0, 0))
        self.assertEqual(buffer.nrow(), 0)
        self.assertEqual(buffer.ncol(), 0)
    
    def test_clear_gpu_buffer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.GPU)
        buffer.clear()
        self.assertEqual(buffer.buffer_size(), 0)
        self.assertEqual(buffer.num_elem(), 0)
        self.assertEqual(buffer.shape(), (0, 0))
        self.assertEqual(buffer.nrow(), 0)
        self.assertEqual(buffer.ncol(), 0)
    
    def test_buffer_string(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        self.assertTrue(repr(buffer).startswith("<Buffer shape=(10, 15) device=CPU @"))
        buffer.to(cuPSO.Device.GPU)
        self.assertTrue(repr(buffer).startswith("<Buffer shape=(10, 15) device=GPU @"))

    def test_buffer_elem_string(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.fill(0.)
        string = str(buffer)
        string = string.replace(',', '')
        self.assertSequenceEqual([float(e) for e in string.split()], [0.] * (10 * 15))
        buffer.to(cuPSO.Device.GPU)
        string = str(buffer)
        string = string.replace(',', '')
        self.assertSequenceEqual([float(e) for e in string.split()], [0.] * (10 * 15))


class TestCURAND(unittest.TestCase):
    def setUp(self):
        print(f"\nTesting: [{__class__.__name__}] {self._testMethodName}  ", end="")

    def test_buffer_size(self):
        states = cuPSO.CURANDStates(100, 0)
        self.assertEqual(states.num_elem(), 100)
        self.assertEqual(states.buffer_size(), 4800)
    
    def test_clear_buffer(self):
        states = cuPSO.CURANDStates(100, 0)
        states.clear()
        self.assertEqual(states.num_elem(), 0)
        self.assertEqual(states.buffer_size(), 0)