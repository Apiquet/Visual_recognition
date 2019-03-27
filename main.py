
from imageai.Detection import ObjectDetection
from IPython.display import Image
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import math

IMAGES_PATH = "images/"
VIDEOS_PATH = "videos/"
LIB_PATH = "imageai_lib/"


execution_path = os.getcwd()

# object creation
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
#loading trained dataset
detector.setModelPath( os.path.join(execution_path , LIB_PATH + "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
#possibility to add argument like car = true to detect only cars
custom_objects = detector.CustomObjects()
#detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "pasteque.jpg"), output_image_path=os.path.join(execution_path , "bottle_detection.png"), minimum_percentage_probability=65)


'''for eachObject in detections:
   print(eachObject["name"] + " : " + str(eachObject["percentage_probability"]) + " : " + str(eachObject["box_points"]) )
   print("--------------------------------")'''


detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , IMAGES_PATH + "pasteque.jpg"), output_image_path=os.path.join(execution_path , IMAGES_PATH+"pasteque_detection.png"), minimum_percentage_probability=40)
#for eachObject in detections:
   #print(eachObject["name"] + " : " + str(eachObject["percentage_probability"]) + " : " + str(eachObject["box_points"]) )
   #print("--------------------------------")
print(IMAGES_PATH + "pasteque_detection.png was created")


custom_objects = detector.CustomObjects(bottle=True)
camera = cv2.VideoCapture(0)
print("Press S key to start object detection on the image")
print("Press Q key to stop the script")
while True:
    return_value,image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',image)
    #object detection
    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite(IMAGES_PATH + 'cam_screenshot.jpg',image)
        detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , IMAGES_PATH + "cam_screenshot.jpg"), output_image_path=os.path.join(execution_path , IMAGES_PATH + "bottle_detection.png"), minimum_percentage_probability=70)
        img_detection=mpimg.imread(IMAGES_PATH + 'bottle_detection.png')
        cv2.imshow(IMAGES_PATH + 'bottle_detection',img_detection)
        print("bottle_detection.png saved")
        #here to access objects detected
        for eachObject in detections:
            print(eachObject["name"] + " : " + str(eachObject["percentage_probability"]) + " : " + str(eachObject["box_points"]) )
            '''object_center = (eachObject["box_points"][0] + eachObject["box_points"][2]) / 2
            bottle_size = math.sqrt((eachObject["box_points"][0]-eachObject["box_points"][2])^2 + (eachObject["box_points"][1]-eachObject["box_points"][3])^2)
            print(str(object_center))
            print(str(bottle_size))'''
    #stop execution
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()


# In[3]:


# code to take screenshot every few seconds

'''import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import time, threading
import math
from PIL import Image

new_width  = 400
new_height = 300

def analysing_stream():
    cv2.imwrite(IMAGES_PATH + 'cam_screenshot.jpg',image)
    img = Image.open(IMAGES_PATH + 'cam_screenshot.jpg') # image extension *.png,*.jpg
    print("------")
    #img = gray
    #img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(IMAGES_PATH + 'resized_screenshot.jpg') 
    print(datetime.datetime.now())
    detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , IMAGES_PATH + "resized_screenshot.jpg"), output_image_path=os.path.join(execution_path , IMAGES_PATH + "bottle_detection.png"), custom_objects=custom_objects, minimum_percentage_probability=25)
    print(datetime.datetime.now())
    img_detection=mpimg.imread(IMAGES_PATH + 'bottle_detection.png')
    cv2.imshow(IMAGES_PATH + 'bottle_detection',img_detection)
    for eachObject in detections:
        print(eachObject["name"] + " : " + str(eachObject["percentage_probability"]) + " : " + str(eachObject["box_points"]) )
        object_center = (eachObject["box_points"][0] + eachObject["box_points"][2]) / 2
        bottle_size = math.sqrt((eachObject["box_points"][0]-eachObject["box_points"][2])^2 + (eachObject["box_points"][1]-eachObject["box_points"][3])^2)
        print(str(object_center))
        print(str(bottle_size))
    threading.Timer(10, analysing_stream).start()



camera = cv2.VideoCapture(0)
return_value,image = camera.read()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',image)
analysing_stream()
while True:
    return_value,image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',image)        
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break       
    if cv2.waitKey(1)& 0xFF == ord('s'):
        print("starting")
        analysing_stream()
        
camera.release()
cv2.destroyAllWindows()'''


# In[2]:


#code to detect object on videos
'''from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, VIDEOS_PATH + "camera_recording.mp4"),
                                output_file_path=os.path.join(execution_path, VIDEOS_PATH + "objects_detection")
                                ,frames_per_second=20, log_progress=True)
print(video_path)'''

