from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
import subprocess
# image specification
img_rows,img_cols,img_depth=16,16,16
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Needed for set_window_handle():
gi.require_version('GstVideo', '1.0')
from gi.repository import GstVideo
dictionary = {0:"circle",	1:"comehere",	2:"cross",	3:"pat",	4:"rightleft",	5:"triangle",	6:"turnaround",	7:"updown",	8:"wave",	9:"z"}
# Training data
X_tr=[]           # variable to store entire dataset
subprocess.call(["cd", "/home/sukrit/OpenNI/Platform/Linux/Bin/x64-Release"])
record=subprocess.Popen(["./Sample-NiHandTracker"],stdout=subprocess.PIPE, shell=True)
for vid in record:
    frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)


    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        plt.imshow(gray, cmap = plt.get_cmap('gray'))
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)

X_tr_array = np.array(X_tr)   # convert the frames read into array

train_data = [X_tr_array]
train_set = np.zeros((1, img_rows,img_cols,img_depth))
train_set[1][0][:][:][:]=X_train[1,:,:,:]

# Pre-processing
train_set = train_set.astype('float32')
train_set -= np.mean(train_set)
train_set /=np.max(train_set)

# Define model
model = Sequential()
model.add(Convolution3D(8,nb_depth=2, nb_row=2, nb_col=2, input_shape=(1,16,16,16), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Convolution3D(16,nb_depth=2, nb_row=2, nb_col=2, activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(50, init='normal', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10,init='normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop')

fname = "weights-Test-CNN2.hdf5"
model.load_weights(fname)

score = model.evaluate(train_set)

pred_probas = score[1]

index=pred_probas.argmax()
guesture=dictionary[index]


def set_frame_handle(bus, message, frame_id):
    if not message.get_structure() is None:
        if message.get_structure().get_name() == 'prepare-window-handle':
            display_frame = message.src
            display_frame.set_property('force-aspect-ratio', True)
            display_frame.set_window_handle(frame_id)

NUMBER_OF_FRAMES = 2 # with more frames than arguments, videos are repeated
relative_height = 1 / float(NUMBER_OF_FRAMES)
window = tkinter.Tk()
window.title("Guesture Recognition using Deep Learning")
window.geometry('500x300')
var=tkinter.StringVar()
label = tkinter.Label( window, textvariable=var)
label.place(x=0, y=100)
label.pack()
Gst.init(None)
def change(v):
    var.set(v)
GObject.threads_init()
display_frame = tkinter.Frame(window, bg='')
display_frame.place(relx = 0, rely = 0.10,
        anchor = tkinter.NW, relwidth = 1, relheight =0.90)
frame_id = display_frame.winfo_id()
player = Gst.ElementFactory.make('playbin', None)
player.set_property('uri',record)
player.set_state(Gst.State.PLAYING)
bus = player.get_bus()
bus.enable_sync_message_emission()
#print("stoping")
##var.set('awqweqweqwe2')
bus.connect('sync-message::element', set_frame_handle, frame_id)
window.after(change,"Class"+index+":"+guesture)
window.mainloop()
