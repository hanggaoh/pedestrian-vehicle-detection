import argparse
from src.detector import ObjectDetector
from src.camera import Camera
from src.utils import display_frame
import cv2

def main(source):
    """
    Main function to run the object detection.
    """
    detector = ObjectDetector()
    camera = Camera(source)

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        results = detector.score_frame(frame)
        frame = detector.plot_boxes(results, frame)

        if not display_frame(frame):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Pedestrian and Vehicle Detection")
    parser.add_argument('--source', type=str, default='0', help='Path to video file or camera index.')
    args = parser.parse_args()
    main(args.source)