import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
#window.geometry('1280x720')
window.configure(background='blue')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System" ,bg="Green"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 

message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=20  ,bg="yellow" ,fg="red",font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="red"  ,bg="yellow"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="yellow"  ,fg="red",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=400)


lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="" ,fg="red"   ,bg="yellow",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

face_classifier=cv2.CascadeClassifier('D:/facerecognition/haar cascade/haarcascade_frontalface_default.xml')
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.35,5)    #vary 3 to 6 minneighbor low value less accuracy
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face=img[y:y+h,x:x+w]
    return cropped_face    
file_name_path='D:/facerecognition/t'
def TakeImages():
    Id=(txt.get())
    name=(txt2.get())  
    #folderName = "user" + Id                                                        # creating the person or user folder when to store images of different faces in different groups
    #folderPath = os.path.join(os.path.dirname(os.path.realpath(file_name_path)),"dataset/"+folderName)
    #if not os.path.exists(folderPath):
        #os.makedirs(folderPath)
 
    if(is_number(Id) and name.isalpha()):
        cap=cv2.VideoCapture(0)
        count=0 #to count no. of sample photos taken of face
        sampleNum = 0
        while True:
            ret,frame=cap.read()
            if face_extractor(frame) is not None:
                count+=1
                sampleNum+=1
                face=cv2.resize(face_extractor(frame),(200,200))
                face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                #file_name_path='D:/facerecognition/test/test'+str(count)+'.jpg'    #folder in pc to store sample images 
                cv2.imwrite('D:/facerecognition/dataset/'+name+"."+ Id + "." + str(sampleNum) + ".jpg",face)   
                #cv2.imwrite( folderpath+ name +".",+ Id + "." + str(sampleNum) + ".jpg",face)                                       #to store or write images in given folder
                cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)           #parameters-face-where we want to be written, (50,50)-coordinate
                cv2.imshow('Face Cropper',face)                                      #to show cropped face thats get storing
            else:
                print("Face not found")
                pass
            if cv2.waitKey(1)==13 or count==100:
                break
        cap.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        print("Images Saved for ID:"+Id+"Name:"+name)
        with open('D:/facerecognition/StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res="Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res="Enter Numeric Id"
            message.configure(text=res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    #print(Ids)
    return faces,Ids

def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("D:/facerecognition/dataset")
    recognizer.train(faces, np.array(Id))        #parameters of train model- image to corresponding label so no need to create seperate folder for diff. faces as id can differentiate among them
    recognizer.save("D:/facerecognition/TrainingImageLabel/Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)
face_classifier=cv2.CascadeClassifier('D:/facerecognition/haar cascade/haarcascade_frontalface_default.xml')
def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.35,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]  #region of interest
        roi=cv2.resize(roi,(200,200))

    return img,roi
def TrackImages():
    face_classifier=cv2.CascadeClassifier('D:/facerecognition/haar cascade/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("D:/facerecognition/TrainingImageLabel/Trainner.yml")    
    df=pd.read_csv("D:/facerecognition/StudentDetails\StudentDetails.csv")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret,frame =cap.read()
        image,face=face_detector(frame)
        try:
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            Id,result=recognizer.predict(face)                                   
            if result<500:
                confidence=int(100*(1-(result)/300))
            if confidence>75:
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                tt=str(Id)+"-"+aa
                cv2.putText(image,str(tt),(250,450),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255))
                cv2.imshow('frame',image)
            else:
                Id='Unknown'
                tt=str(Id)
                #noOfFile=len(os.listdir("D:/facerecognition/ImagesUnknown"))+1
                #cv2.imwrite("D:/facerecognition/ImagesUnknown\Image"+str(noOfFile) + ".jpg", image[y:y+h,x:x+w])            
                cv2.putText(image,"unknown",(250,450),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255))          
                cv2.imshow('frame',image)
        
        except:
            cv2.putText(image,"Face not Found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(250,125,255))
            cv2.imshow('frame',image)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        if cv2.waitKey(1)==13:
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="D:/facerecognition/Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cap.release()
    cv2.destroyAllWindows()
    print(attendance)
    res=attendance
    message2.configure(text= res)




clearButton = tk.Button(window, text="Clear", command=clear  ,fg="red"  ,bg="yellow"  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="red"  ,bg="yellow"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.configure(state="disabled",fg="red"  )
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)
