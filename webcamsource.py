import cv2
from PIL import Image
import numpy as np

class WebcamSource():
    def __init__(self, camera_id=0, fps=30, display=False):
        self.__name = "WebcamSource"
        self.__capture = cv2.VideoCapture(camera_id)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.__capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.__capture.set(cv2.CAP_PROP_FPS, fps)
        self.__display = display

    def set_process(self, process):
        self.__process = process

    def __iter__(self):
        self.__capture.isOpened()
        return self

    def __next__(self):
        ret, frame = self.__capture.read()
        
        if self.__display:
            cv2.imshow(f'{self.__name} - FPS: {fps}', frame)

        if not ret:
            raise StopIteration

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise StopIteration

        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb))

        return frame, frame_rgb

    def __del__(self):
        self.__capture.release()
        cv2.destroyAllWindows()

    def show(self, frame):
        fps = self.__capture.get(cv2.CAP_PROP_FPS)
        cv2.imshow(f'{self.__name} - FPS: {fps}', frame)
