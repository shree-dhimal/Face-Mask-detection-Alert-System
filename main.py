############################################# IMPORTING ################################################
from asyncio.windows_events import NULL
from email import message
import email
from sqlite3 import Timestamp
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
from turtle import delay
from winsound import PlaySound
import cv2,os
from cv2 import imwrite
from cv2 import VideoCapture
from matplotlib import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
from matplotlib.pyplot import text
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import re
import mysql.connector
from tkinter import messagebox
import smtplib , ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from mtcnn.mtcnn import MTCNN
from threading import Event

#t_end = time.time() + 15 * 1
#t_ending = time.time() + 10 * 1
#####################################################################################################

from pygame import mixer
mixer.init()
sound = mixer.Sound('Alarm/santosh.wav')


#############################################################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

###################################################################################

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'shreedhimal1@gmail.com' ")

###################################################################################

def check_facemasktrainmodel():
    exists = os.path.isfile("mask_detector.model")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

#####################################################################################################
def delete_rec():
    sucess = False
    delete_id = remove_id.get()
    if (remove_id.get()==""):
        messagebox.showerror("Error","Please Enter ID to remove")
    elif delete_id.isnumeric:
        mydb = mysql.connector.connect(
            host ="localhost",
            user = "root",
            passwd = "",
            database = "Face_mask_detection_alert_system"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT Count(*) from staff WHERE ID="+str(delete_id))
        myresult = mycursor.fetchone()
        
        if myresult != 0:
            sql = "DELETE FROM staff WHERE ID="+ str(delete_id)
            mycursor.execute(sql)
            mydb.commit()

            #removing images

            assure_path_exists("TrainingImage/")
            path = "TrainingImage/"
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            for imagePath in imagePaths:
                ID = int(os.path.split(imagePath)[-1].split(".")[1])
                del_id = int(delete_id)
                if del_id  == ID:
                    os.remove(os.path.join(imagePath))
                    sucess = True
                else:
                    sucess = False
            if sucess == True:
                messagebox.showinfo("Warning","Sucessfully Removed data of ID:"+delete_id)
                TrainImages()
            else:
                messagebox.showerror("Error","cannot find data of ID:"+delete_id)
        else:
            messagebox.showinfo("Warning"," No data found in database for the Entered ID")
        
    else:
        messagebox.showinfo("Warning","Please Enter valid ID")
    initilizer()
    clear1()
    
        



###################################################################################

def change_email():
    global master
    master = tk.Tk()
    master.geometry("400x170")
    master.resizable(False,False)
    master.title("Change Email")
    master.configure(background="white")
    lbl5 = tk.Label(master, text='Enter new Email', bg='white', font=('comic', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black",relief='solid', font=('comic', 12, ' bold '))
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Email', bg='white', font=('comic', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid',font=('comic', 12, ' bold '))
    nnew.place(x=180, y=80)
    cancel=tk.Button(master,text="Cancel", command=master.destroy ,fg="black"  ,bg="red" ,height=1,width=25 , activebackground = "white" ,font=('comic', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_email, fg="black", bg="#00fcca", height = 1,width=25, activebackground="white", font=('comic', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()


def save_email():
    newp= (new.get())
    nnewp = (nnew.get())
    if  (newp !='') or (nnewp !=''):
        if (newp == nnewp):
            if (re.fullmatch(regex, nnewp)) and (re.fullmatch(regex, newp)):
                mydb = mysql.connector.connect(
                host ="localhost",
                user = "root",
                passwd = "",
                database = "face_mask_detection_alert_system"
                )
                mycursor = mydb.cursor()
                mycursor.execute("SELECT * from Admin where id=1")
                myresult = mycursor.fetchone()
                
                sql = """ UPDATE Admin
                        SET email = %s
                        WHERE id = %s """
                mail = (nnewp,1)
                mycursor.execute(sql,mail)
                mydb.commit()
                mydb.close()
                messagebox.showinfo("Sucess","Sucessfully Changed Admin Email!!!")
            else:
                messagebox.showerror("Error","Enter Valid email address!!!")
                master.destroy()
        else:
            messagebox.showerror("Error","confirm new email again")
        master.destroy()
    else:
        messagebox.showerror("Error","Dont Leave any field empty")
        master.destroy()


#####################################################################################
#clear buttons

def clear2():
    txt2.delete(0, 'end')
    
def clear3():
    staff_email.delete(0, 'end')

def clear1():
    remove_id.delete(0,'end')

#######################################################################################
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


########################################################################################



def TakeImages():
    assure_path_exists("TrainingImage/")
    serial = 0
    Id = int(txt.cget("text"))
    name = (txt2.get())
    email = (staff_email.get())

    if name == '' or email == '':
        messagebox.showerror("Error","Dont leave the field empty")
    else:
        if ((name.isalpha()) or (' ' in name)) and (re.fullmatch(regex, email)):

            cam = cv2.VideoCapture(0)
            face_classifier = MTCNN()
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                
                faces = face_classifier.detect_faces(img)
                if len(faces) > 0:
                    for face in faces:
                        x, y, width, height = face['box']
                        x2, y2 = x + width, y + height

                        x1, y1, width, height = faces[0]['box']
                        x3, y3 = x1 + width , y1+ height
                        face_save = img[y1:y3,x1:x3]
                        gray = cv2.cvtColor(face_save, cv2.COLOR_BGR2GRAY)

                        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 4)
                    
                        sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder TrainingImage
                        cv2.imwrite("TrainingImage/" + name + "." + str(Id) + '.' + str(sampleNum) + ".jpg", gray)
                    # display the frame
                        cv2.imshow('Taking Images', img)
                # wait for 50 miliseconds
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 50
                elif sampleNum > 50:
                    break
                
            cam.release()
            cv2.destroyAllWindows()

            mydb = mysql.connector.connect(
                host ="localhost",
                user = "root",
                passwd = "",
                database = "face_mask_detection_alert_system"
            )
            mycursor = mydb.cursor()
            mycursor.execute("SELECT * from staff")
            myresult = mycursor.fetchall()
            
            sql = "insert into staff(id, name , email) values(%s,%s,%s)"
            val = (int(Id), str(name) , email)
            mycursor.execute(sql,val)
            mydb.commit()

            TrainImages()
            initilizer()
        else:
            if (name.isalpha() == False):
                messagebox.showerror("Error","Enter Correct name!!!")
            if re.fullmatch(regex, email) is None:
                messagebox.showerror("Error","Enter Valid email address!!!")
     

########################################################################################

#train the caputred image

def TrainImages():
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, ID = getImagesAndLabels("TrainingImage/")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        #messagebox.showerror(title='zero data', message='Unsucessfull')
        return
    recognizer.save("TrainingImageLabel\Trainner.xml")

    messagebox.showinfo("Warning","Successfully")
    
    initilizer()
    clear2()
    clear3()

############################################################################################3

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

###########################################################################################

#face mask model
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

#run the model to check whether masked or unmasked
def detect_facemask():
    check_facemasktrainmodel()
    
    label = ''
    PlaySound = True
    # load our caffe serialized face detector model from disk
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    vs = VideoStream(src=0).start()
    #if vs is None:
      # messagebox.showerror("ERROR","Cant Open the Camera Please Ensure the camera is Running!!")
       #return None


    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()

        frame = imutils.resize(frame, width=800)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            if label == "No Mask": 

                frame2 = vs.read()
                sound.play()
                
                detect_unmasked_image(frame2)
                continue
            else:
                
                continue

        # show the osutput frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
               

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    
#######################################################################################################

#if unmasked run this model
def detect_unmasked_image(frame2):
    
    face_classifier = MTCNN()
    faces = face_classifier.detect_faces(frame2)
    if faces is ():
        return None
    else:
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        
        assure_path_exists("unmaskedimages/")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("TrainingImageLabel\Trainner.xml")

        if len(faces) > 0:
            for face in faces:
                x1, y1, width, height = faces[0]['box']
                x3, y3 = x1 + width , y1+ height
                
                face_save = frame2[y1:y3,x1:x3]
                
                gray = cv2.cvtColor(face_save, cv2.COLOR_BGR2GRAY)

                id, pred = clf.predict(gray)
                
                confidence = int(100*(1-pred/300))
                
                print(pred)
                print(id)
                print(confidence)
                
                if (confidence > 80):
                    mydb = mysql.connector.connect(
                    host ="localhost",
                    user = "root",
                    passwd = "",
                    database = "face_mask_detection_alert_system"
                    )
                    mycursor = mydb.cursor()
                    mycursor.execute("SELECT Id, name , Email from staff where id="+ str(id))
                    myresult = mycursor.fetchone()
                    unmasked_ID = myresult[0]
                    unmasked_name = myresult[1]
                    unmasked_email = myresult[2]

                    final_id = unmasked_ID
                    final_name = unmasked_name
                    final_email = unmasked_email

                else:
                    final_id = '0'
                    final_name ='Visitor'
                    final_email ='0'
                
            image1 = cv2.imwrite("unmaskedimages\ "+ final_name +"."+ str(final_id)+"."+str(ts)+".png",frame2)
            if image1 is True:
                image_path = "unmaskedimages/ "+ final_name +"."+ str(final_id)+"."+str(ts)+".png"

            print("sucess")
            print(final_id)
            print(final_name)
            print(final_email)
            print(Timestamp)
            print(date)

            database_entry(final_id, final_name , final_email, date, timeStamp, image_path)
            
#entry data into the database 
def database_entry(final_id, final_name , final_email, date, timeStamp, frame):

    if final_email is ("0"):
        mydb = mysql.connector.connect(
        host ="localhost",
        user = "root",
        passwd = "",
        database = "face_mask_detection_alert_system"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT email from Admin where id = 1")
        myresult = mycursor.fetchall()
        admin_email = myresult[0][0]
        print(admin_email)
        main_final_email = admin_email
    else:
        main_final_email = final_email


    mydb = mysql.connector.connect(
        host ="localhost",
        user = "root",
        passwd = "",
        database = "face_mask_detection_alert_system"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * from entry")
    myresult = mycursor.fetchall()
    sql = "insert into entry(id,name,date,time) values(%s, %s, %s, %s)"
    val = (final_id, final_name, date, timeStamp)
    mycursor.execute(sql,val)
    mydb.commit()
    
    if final_name=='Visitor':
        content = "Visitor Entered Without Mask"
    else:
        content = final_name+" "+"Please Wear a mask!!"
    print("sucess")
    alert_email(main_final_email, content, frame)

#sent email and image to mail
def alert_email(email, content, frame):

    #email id and password
    sender = 'noreplynofacemaskalert@gmail.com'
    sender_pass = '9843393667'

    reciever = email

    img = frame
    msg = MIMEMultipart()
    msg['TO'] = reciever
    msg['From'] = sender
    msg['Subject'] = 'Alert'+'<'+sender+'>'

    msg_ready = MIMEText(content)

    image_open = open(img,'rb').read()

    image_ready = MIMEImage(image_open,'png', name='unmasked.png')

    msg.attach(msg_ready)
    
    msg.attach(image_ready)

    context_data = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context_data) as mail:
        mail.login(sender, sender_pass)
        mail.sendmail(sender,reciever,msg.as_string())
    for k in tv.get_children():
        tv.delete(k)
    add_data_grid()

    
###############################################################################################
#refresh datagrid view
def add_data_grid():
    i=0
    j=0
    cpt= 0
    mydb = mysql.connector.connect(
        host ="localhost",
        user = "root",
        passwd = "",
        database = "face_mask_detection_alert_system"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * from entry")
    myresult = mycursor.fetchall()
    if myresult is NULL:
        return None
    else:
        for lines in myresult:
            id= lines[0]
            name = lines[1]
            date = lines[2]
            time = lines[3]
            tv.insert('','end',text=str(cpt),values=(str(id),str(name),str(date),str(time)))
            cpt += 1


###################################################################################################
#clear image, records form database and gridview
def clear_columns():

    mydb = mysql.connector.connect(
        host ="localhost",
        user = "root",
        passwd = "",
        database = "Face_mask_detection_alert_system"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT count(id) from entry")
    myresult = mycursor.fetchone()
    print(myresult)
    if myresult[0] is NULL:
        messagebox.showinfo("Warning","No rows to Delete")
    else:
        sql = ("DELETE FROM entry")
        mycursor.execute(sql)
        mydb.commit()

            #removing images
        assure_path_exists("unmaskedimages/")
        path = "unmaskedimages/"
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        if imagePaths is ():
            return None
        else:
            for imagePath in imagePaths:
                os.remove(os.path.join(imagePath))
        for k in tv.get_children():
            tv.delete(k)
        add_data_grid()
        messagebox.showinfo("Sucess","Sucessfully Removed all Rows")
        



######################################## USED STUFFS ############################################
    
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }

######################################## GUI ###########################################

window = tk.Tk()
fullScreenState =False
width = window.winfo_screenwidth()
height = window.winfo_screenheight()

print(width)
print(height)
#window.attributes('-fullscreen', True)
window.geometry("%dx%d" % (width, height))
window.state('zoomed')
window.resizable(False,True)
window.title("Face Mask Alert System")



window.configure(background='#2d420a')

frame1 = tk.Frame(window, bg="#c79cff")
frame1.place(relx=0.05, rely=0.17, relwidth=0.45, relheight=0.80)

frame2 = tk.Frame(window, bg="#c79cff")
frame2.place(relx=0.55, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face Mask Detection Alert System" ,fg="white",bg="#2d420a" ,width=55 ,height=1,font=('comic', 29, ' bold '))
message3.place(x=10, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text = day+"-"+mont[month]+"-"+year+"  |  ", fg="#ff61e5",bg="#2d420a" ,width=55 ,height=1,font=('comic', 22, ' bold '))
datef.pack(fill='both',expand=1)

clock = tk.Label(frame3,fg="#ff61e5",bg="#2d420a" ,width=55 ,height=1,font=('comic', 22, ' bold '))
clock.pack(fill='both',expand=1)
tick()

head2 = tk.Label(frame2, text="                        New Registrations                       ", fg="black",bg="#00fcca" ,font=('comic', 17, ' bold ') )
head2.grid(row=0,column=0)

lbl = tk.Label(frame2, text="ID:",width=0  ,height=1  ,fg="black"  ,bg="#c79cff" ,font=('comic', 17, ' bold ') )
lbl.place(x=80, y=60)

txt = tk.Label(frame2, text="", width=0,height=1  ,fg="black"  ,bg="#c79cff" ,font=('comic', 17, ' bold '))
txt.place(x=150, y=60)

lbl2 = tk.Label(frame2, text="Enter Name",width=20  ,fg="Black"  ,bg="#c79cff" ,font=('comic', 17, ' bold '))
lbl2.place(x=80, y=135)

txt2 = tk.Entry(frame2,width=32 ,fg="Green",font=('comic', 15, ' bold ')  )
txt2.place(x=30, y=168)

lbl3 = tk.Label(frame2, text="Enter Email",width=20  ,fg="black"  ,bg="#c79cff" ,font=('comic', 17, ' bold '))
lbl3.place(x=80, y=220)
staff_email = tk.Entry(frame2,width=32 ,fg="Green",font=('comic', 15, ' bold ')  )
staff_email.place(x=30, y=253)


lbl4 = tk.Label(frame2, text="Enter ID to remove",width=20  ,fg="black"  ,bg="#c79cff" ,font=('comic', 17, ' bold '))
lbl4.place(x=30, y=370)
remove_id = tk.Entry(frame2,width=32 ,fg="Green",font=('comic', 15, ' bold ')  )
remove_id.place(x=30, y=400)


lbl3 = tk.Label(frame1, text="People With no mask",width=20  ,fg="black"  ,bg="#c79cff"  ,height=1 ,font=('comic', 17, ' bold '))
lbl3.place(x=150, y=115)

#######################################################################################################

#initilize the Staff ID from the database
def initilizer():
    #total_int = int ()
    mydb = mysql.connector.connect(
    host ="localhost",
    user = "root",
    passwd = "",
    database = "face_mask_detection_alert_system"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id) FROM staff")
    total_ids = mycursor.fetchone()
    
    if total_ids[0] is None:
       total_int =1
    else:
        total_int = total_ids[0]
        total_int = total_int + 1
    txt.configure(text=total_int)

initilizer()



##################### MENUBAR #################################

menubar = tk.Menu(window,relief='ridge')
filemenu = tk.Menu(menubar,tearoff=0)
filemenu.add_command(label='Change Email', command = change_email)
filemenu.add_command(label='Contact Us', command = contact)
filemenu.add_command(label='Exit',command = window.destroy)
menubar.add_cascade(label='Help',font=('comic', 29, ' bold '),menu=filemenu)

################## TREEVIEW ATTENDANCE TABLE ####################

tv= ttk.Treeview(frame1,height =13,columns = ('id','name','date','time'))
tv.column('#0',width=70)
tv.column('id',width=100)
tv.column('name',width=144)
tv.column('date',width=144)
tv.column('time',width=144)
tv.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tv.heading('#0',text ='S.N')
tv.heading('id',text ='ID')
tv.heading('name',text ='NAME')
tv.heading('date',text ='Date')
tv.heading('time',text ='Time')


###################### SCROLLBAR ################################

scroll=ttk.Scrollbar(frame1,orient='vertical',command=tv.yview)
scroll.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
tv.configure(yscrollcommand=scroll.set)

##################################################################

add_data_grid()

###################### BUTTONS ##################################

clearButton = tk.Button(frame2, text="Clear", command=clear1  ,fg="black"  ,bg="#ff7221"  ,width=11 ,activebackground = "white" ,font=('comic', 11, ' bold '))
clearButton.place(x=335, y=400)
clearButton2 = tk.Button(frame2, text="Clear", command=clear2  ,fg="black"  ,bg="#ff7221"  ,width=11 , activebackground = "white" ,font=('comic', 11, ' bold '))
clearButton2.place(x=335, y=167)   
clearButton3 = tk.Button(frame2, text="Clear", command=clear3  ,fg="black"  ,bg="#ff7221"  ,width=11 , activebackground = "white" ,font=('comic', 11, ' bold '))
clearButton3.place(x=335, y=253) 
takeImg = tk.Button(frame2, text="Take Images", command=TakeImages  ,fg="white"  ,bg="#6d00fc"  ,width=34  ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
takeImg.place(x=30, y=295)
trainImg = tk.Button(frame2, text="Remove", command=delete_rec ,fg="white"  ,bg="#6d00fc"  ,width=34  ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
trainImg.place(x=30, y=433)
trackImg = tk.Button(frame1, text="Run Program", command=detect_facemask  ,fg="black"  ,bg="#3ffc00"  ,width=35  ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
trackImg.place(x=30,y=50)
#detect_facemask
quitWindow = tk.Button(frame1, text="Quit", command=window.destroy  ,fg="black"  ,bg="#eb4600"  ,width=35 ,height=1, activebackground = "white" ,font=('comic', 15, ' bold '))
quitWindow.place(x=30, y=450)
quitWindow = tk.Button(frame1, text="Clear Columns", command=clear_columns  ,fg="black"  ,bg="#eb4600"  ,width=15 ,height=1, activebackground = "white" ,font=('comic', 12, ' bold '))
quitWindow.place(x=450, y=115)
##################### END ######################################

window.configure(menu=menubar)
window.mainloop()

####################################################################################################
