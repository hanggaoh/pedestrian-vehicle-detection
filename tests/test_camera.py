import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Since the src directory is not in the python path by default, we need to add it
import sys
sys.path.append('.')

from src.camera import Camera

class TestCamera(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_get_frame(self, mock_video_capture):
        """
        Test the get_frame method.
        """
        mock_capture_instance = MagicMock()
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_capture_instance

        camera = Camera(0)
        frame = camera.get_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

    @patch('cv2.VideoCapture')
    def test_release(self, mock_video_capture):
        """
        Test the release method.
        """
        mock_capture_instance = MagicMock()
        mock_video_capture.return_value = mock_capture_instance

        camera = Camera(0)
        camera.release()

        mock_capture_instance.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()
