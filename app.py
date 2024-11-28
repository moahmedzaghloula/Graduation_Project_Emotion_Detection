<<<<<<< HEAD
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Cache the model loading function using st.cache_resource
@st.cache_resource
def load_emotion_model():
    # Load the model architecture
    json_file = open('emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # Load the model weights
    model = model_from_json(loaded_model_json)
    model.load_weights("emotion_model.weights.h5")
    return model

# Load the pre-trained model
model = load_emotion_model()

# Load the face detector from OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the Streamlit app
st.title("Real-time Emotion Detection")

# Start capturing video from webcam or allow user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

if uploaded_file is not None:
    # If an image is uploaded, use it for emotion detection
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
else:
    st.text("Waiting for you to upload an image...")

# Function to perform emotion detection on an image
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Display the emotion on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame

if uploaded_file is not None:
    # Detect emotions and display the result
    result_frame = detect_emotion(image)
    st.image(result_frame, channels="BGR")
=======
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

# Cache the model loading function using st.cache_resource
@st.cache_resource
def load_emotion_model():
    # Load the model architecture
    json_file = open('emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # Load the model weights
    model = model_from_json(loaded_model_json)
    model.load_weights("emotion_model.weights.h5")
    return model

# Load the pre-trained model
model = load_emotion_model()

# Load the face detector from OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the Streamlit app
st.title("Real-time Emotion Detection")

# Start capturing video from webcam or allow user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

if uploaded_file is not None:
    # If an image is uploaded, use it for emotion detection
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
else:
    st.text("Waiting for you to upload an image...")

# Function to perform emotion detection on an image
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Display the emotion on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame

if uploaded_file is not None:
    # Detect emotions and display the result
    result_frame = detect_emotion(image)
    st.image(result_frame, channels="BGR")
>>>>>>> 84d1c3af3cdeedd0fff36ab347655fc7f672b3be
