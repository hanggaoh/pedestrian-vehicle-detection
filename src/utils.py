import cv2

def display_frame(frame):
    """
    Displays the frame.
    """
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True
