import cv2

class Camera:
    def __init__(self, source):
        self.source = source
        try:
            self.source = int(source)
        except ValueError:
            pass
        self.capture = cv2.VideoCapture(self.source)

    def get_frame(self):
        """
        Reads a frame from the camera source.
        :return: A single frame.
        """
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def release(self):
        """
        Releases the camera source.
        """
        self.capture.release()
