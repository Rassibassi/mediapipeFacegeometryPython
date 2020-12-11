import numpy as np
from PIL import Image
import mediapipe as mp

def load_image(infilename, dtype="uint8"):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype=dtype)
    return data

image = load_image("frame.jpeg")
frame_height, frame_width, channels = image.shape
print(image.shape)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
results = face_mesh.process(image)
face_landmarks = results.multi_face_landmarks[0]
landmarks = np.array([(lm.x,lm.y,lm.z,lm.visibility,lm.presence) for lm in face_landmarks.landmark])
landmarks = landmarks.T
landmarks = landmarks[:3, :]
landmarks = landmarks[:3, :]

np.save('landmarks.npy', landmarks)