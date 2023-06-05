import torch
import cv2

class ObjectDetector:
    def __init__(self, model_path='weights/yolov5s.pt'):
        """
        Initializes the object detector with the YOLOv5 model.
        """
        self.model = self._load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def _load_model(self, model_path):
        """
        Loads the YOLOv5 model from the specified path.
        """
        # Hub model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using the YOLOv5 model.
        :param frame: input frame in numpy array format.
        :return: labels and coordinates of objects detected.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given tensor x, returns the corresponding class label.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and results as input, and plots the bounding boxes and labels on the frame.
        :param results: labels and coordinates predicted by model
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels plotted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
