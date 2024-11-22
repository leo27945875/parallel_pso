import cuPSO
import unittest
import numpy as np


class TestBuffer(unittest.TestCase):
    def test_buffer_size(self):
        buffer = cuPSO.Buffer(4, 3, cuPSO.Device.CPU)
        self.assertEqual(buffer.buffer_size(), 12 * 8)
        self.assertEqual(buffer.num_elem(), 12)
        self.assertEqual(buffer.shape(), (4, 3))
        self.assertEqual(buffer.nrow(), 4)
        self.assertEqual(buffer.ncol(), 3)
        self.assertEqual(buffer.device(), cuPSO.Device.CPU)
    
    def test_device_definitions(self):
        self.assertEqual(cuPSO.Device.CPU.name, "CPU")
        self.assertEqual(cuPSO.Device.GPU.name, "GPU")
        self.assertEqual(cuPSO.Device.CPU.value, 0)
        self.assertEqual(cuPSO.Device.GPU.value, 1)

    def test_cpu_buffer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        buffer.fill(5.)
        self.assertEqual(buffer[2, 3], 5.)
        self.assertEqual(buffer[9, 14], 5.)
    
    def test_gpu_buffer(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.GPU)
        buffer.fill(5.)
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
    
    def test_buffer_string(self):
        buffer = cuPSO.Buffer(10, 15, cuPSO.Device.CPU)
        self.assertTrue(str(buffer).startswith("<Buffer shape=(10, 15) device=CPU @"))
        buffer.to(cuPSO.Device.GPU)
        self.assertTrue(str(buffer).startswith("<Buffer shape=(10, 15) device=GPU @"))