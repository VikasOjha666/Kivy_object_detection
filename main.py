from kivy.lang import Builder
from kivy.properties import ObjectProperty

from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivymd.uix.screen import MDScreen
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np


#Initializing some important constants.

with open('classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
scale = 0.00392
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
det_model=cv2.dnn.readNet('./yolov4-tiny.weights','./tiny_yolov4.cfg')



def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def detect_on_image(image):
    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    det_model.setInput(blob)
    outs = det_model.forward(get_output_layers(det_model)) # forward function

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        x1=round(x)
        y1=round(y)
        x2=round(x+w)
        y2=round(y+h)
        label = str(classes[i])
        color = COLORS[i]
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 3)
        cv2.putText(image, label,(int(x1), int(y1-10)),0, 0.75, (255,255,255),2)
        
    return image


class CameraScreen(MDScreen):
    def __init__(self, **kwargs): 
        super(CameraScreen, self).__init__(**kwargs)
        self.layout=MDBoxLayout(orientation="vertical")
        self.image=Image()
        self.layout.add_widget(self.image)
        self.add_widget(self.layout)

        self.vid=cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video,1.0/20)

    def load_video(self,*args):
        ret,img=self.vid.read()
        img=cv2.flip(img,0)
        img=detect_on_image(img)
        buffer=img.tostring()
        texture=Texture.create(size=(img.shape[1],img.shape[0]),colorfmt='bgr')
        texture.blit_buffer(buffer,colorfmt='bgr',bufferfmt='ubyte')
        self.image.texture=texture
class ContentNavigationDrawer(MDBoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()
    

class main(MDApp):
    pass

main().run()