from kivy.app import App
import cv2

class MainApp(App):
    def build(self):
        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        classNames = []
        classFile = 'coco.names'
        with open(classFile,'rt') as f:
            #hepsini classnames a at
            classNames = f.read().rstrip("\n").split("\n")
        
        # yapılandırma yolu
        configPath = "uzuncoco.pbtxt"
        weightsPath = "frozen_inference_graph.pb"
        #modelimizi oluşturuyoruz
        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/127.5)
        net.setInputMean((127.5,127.5,127.5))
        net.setInputSwapRB(True)
        
        
  
        
        while True:
            success,img = cap.read()
            #değer 50 üzerindeyse
            classIds, confs, bbox = net.detect(img,0.5)  
            print(classIds,bbox)
            
            for classId,confidence,box in  zip(classIds.flatten(),confs.flatten(),bbox):
                # dikdörtgen
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                # metin
                cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                
            cv2.imshow("output",img)
            cv2.waitKey(1)

if __name__== "__main__":
    MainApp().run()