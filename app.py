import numpy as np
import streamlit as st
import cv2
from deepface import DeepFace as dfc
from PIL import Image
import os
import time


st.set_page_config( 
layout="wide",  
initial_sidebar_state="auto",
page_title= "Face Detection and Analyzer",
)

# function to load image
try:
    face_cascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_alt2.xml')
except Exception:
    st.write("Error loading cascade classifiers")

@st.cache
def face_detect(img):
    img = np.array(img.convert("RGB"))
    face = face_cascade.detectMultiScale(image=img)

    # draw rectangle around face
    for (x, y, w, h) in face:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi = img[y:y + h, x:x + w]
    return img, face

# analyze image
def analyze_image(img):
    prediction = dfc.analyze(img_path=img)
    return prediction

#function for webcam
def detect_web(image):

    faces = face_cascade.detectMultiScale(
        image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)

    return image, faces


def main():
    # Face Analysis Application #
    st.title("Face Detection and Analysis Application")
    activiteis = ["Home", "Analyze Face From File", "Analyze Face From Webcam", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    # C0C0C0
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face detection and Face feature analysis application using OpenCV, DeepFace and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        
    elif choice == "Analyze Face From File":
        st.subheader("Analyze facial features such as emotion, age, gender and race.")
        image_file = st.file_uploader("Upload image you want to analyze", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            #read image using PIL
            image_loaded = Image.open(image_file)
            #detect faces in image
            result_img, result_face = face_detect(image_loaded)
            st.image(result_img, use_column_width=True)
            st.success("found {} face\n".format(len(result_face)))

            if st.button("Analyze image"):
                # convert image to array
                new_image = np.array(image_loaded.convert('RGB'))
                img = cv2.cvtColor(new_image, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                #analyze features of face
                result = analyze_image(img)
                # st.write(result)
                st.write("Analysis summary")
                st.write("your face emotion is ", result["dominant_emotion"], "in image.")
                st.write("Gender recognized as", result["gender"], "in image.")
                st.write("Your age is", result["age"], "years.")
                st.write("It look like you belongs to", result["dominant_race"], "race.")
            else:
                pass
                #st.write("Click on Analyze image ")
                
                
    elif choice == "Analyze Face From Webcam":
        st.subheader("Analyze facial features such as emotion, age, gender and race.")
        image_file = st.camera_input("Take a picture")

        if image_file is not None:
            #read image using PIL
            image_loaded = Image.open(image_file)
            #detect faces in image
            result_img, result_face = face_detect(image_loaded)
            st.image(result_img, use_column_width=True)
            st.success("found {} face\n".format(len(result_face)))

            if st.button("Analyze image"):
                # convert image to array
                new_image = np.array(image_loaded.convert('RGB'))
                img = cv2.cvtColor(new_image, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                #analyze features of face
                result = analyze_image(img)
                # st.write(result)
                st.write("Analysis summary")
                st.write("your face emotion is ", result["dominant_emotion"], "in image.")
                st.write("Gender recognized as", result["gender"], "in image.")
                st.write("Your age is", result["age"], "years.")
                st.write("It look like you belongs to", result["dominant_race"], "race.")
            else:
                pass


    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        run = st.checkbox('Run')
        time.sleep(2.0)
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        while run:
            _, img = camera.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, a = detect_web(img)
            FRAME_WINDOW.image(img)

    else:
        pass

if __name__ == '__main__':
    main()