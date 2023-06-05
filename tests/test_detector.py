import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Since the src directory is not in the python path by default, we need to add it
import sys
sys.path.append('.')

from src.detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):

    @patch('torch.hub.load')
    def test_load_model(self, mock_torch_hub_load):
        """
        Test that the model loads correctly.
        """
        mock_model = MagicMock()
        mock_model.names = ['person', 'car']
        mock_torch_hub_load.return_value = mock_model

        detector = ObjectDetector()
        self.assertIsNotNone(detector.model)
        mock_torch_hub_load.assert_called_with('ultralytics/yolov5', 'custom', path='weights/yolov5s.pt')

    @patch('torch.hub.load')
    def test_score_frame(self, mock_torch_hub_load):
        """
        Test the score_frame method.
        """
        mock_model = MagicMock()
        mock_model.names = ['person', 'car']
        # Simulate a detection
        mock_results = MagicMock()
        mock_results.xyxyn = [torch.tensor([[0.1, 0.1, 0.5, 0.5, 0.9, 0]])] # x1, y1, x2, y2, conf, class
        mock_model.return_value = mock_results
        mock_torch_hub_load.return_value = mock_model

        detector = ObjectDetector()
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        labels, cord = detector.score_frame(frame)

        self.assertEqual(len(labels), 1)
        self.assertEqual(len(cord), 1)

    @patch('torch.hub.load')
    def test_class_to_label(self, mock_torch_hub_load):
        """
        Test the class_to_label method.
        """
        mock_model = MagicMock()
        mock_model.names = ['person', 'car']
        mock_torch_hub_load.return_value = mock_model

        detector = ObjectDetector()
        self.assertEqual(detector.class_to_label(0), 'person')
        self.assertEqual(detector.class_to_label(1), 'car')

if __name__ == '__main__':
    unittest.main()
