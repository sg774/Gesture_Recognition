from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras import regularizers
import keras.models
from keras.models import Sequential
import theano
import os
import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
#theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

#from keras import backend as K
#K.set_image_dim_ordering('th')
os.environ['KERAS_BACKEND']='theano'
# image specification
img_rows,img_cols,img_depth=16,16,16

# Training data
X_tr=[]           # variable to store entire dataset


# In[18]:

#Reading circle gesture class
listing = os.listdir('/home/ubuntu/dataset_depth/circle')
for vid in listing:
    vid = '/home/ubuntu/dataset_depth/circle/'+vid
    frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)


    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[19]:

#Reading comehere gesture class

listing2 = os.listdir('/home/ubuntu/dataset_depth/comehere')
for vid2 in listing2:
    vid2 = '/home/ubuntu/dataset_depth/comehere/'+vid2
    frames = []
    cap = cv2.VideoCapture(vid2)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[20]:

#Reading cross gesture class

listing3 = os.listdir('/home/ubuntu/dataset_depth/cross')
for vid3 in listing3:
    vid3 = '/home/ubuntu/dataset_depth/cross/'+vid3
    frames = []
    cap = cv2.VideoCapture(vid3)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[21]:

#Reading pat gesture class

listing4 = os.listdir('/home/ubuntu/dataset_depth/pat')
for vid4 in listing4:
    vid4 = '/home/ubuntu/dataset_depth/pat/'+vid4
    frames = []
    cap = cv2.VideoCapture(vid4)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[22]:

#Reading rightleft gesture class

listing5 = os.listdir('/home/ubuntu/dataset_depth/rightleft')
for vid5 in listing5:
    vid5 = '/home/ubuntu/dataset_depth/rightleft/'+vid5
    frames = []
    cap = cv2.VideoCapture(vid5)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)



# In[23]:

#Reading triangle gesture class

listing6 = os.listdir('/home/ubuntu/dataset_depth/triangle')
for vid6 in listing6:
    vid6 = '/home/ubuntu/dataset_depth/triangle/'+vid6
    frames = []
    cap = cv2.VideoCapture(vid6)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[24]:

#Reading turaround gesture class

listing7 = os.listdir('/home/ubuntu/dataset_depth/turnaround')
for vid7 in listing7:
    vid7 = '/home/ubuntu/dataset_depth/turnaround/'+vid7
    frames = []
    cap = cv2.VideoCapture(vid7)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[25]:

#Reading updown gesture class

listing8 = os.listdir('/home/ubuntu/dataset_depth/updown')
for vid8 in listing8:
    vid8 = '/home/ubuntu/dataset_depth/updown/'+vid8
    frames = []
    cap = cv2.VideoCapture(vid8)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[26]:

#Reading wave gesture class

listing9 = os.listdir('/home/ubuntu/dataset_depth/wave')
for vid9 in listing9:
    vid9 = '/home/ubuntu/dataset_depth/wave/'+vid9
    frames = []
    cap = cv2.VideoCapture(vid9)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[27]:

#Reading z gesture class

listing10 = os.listdir('/home/ubuntu/dataset_depth/z')
for vid10 in listing10:
    vid10 = '/home/ubuntu/dataset_depth/z/'+vid10
    frames = []
    cap = cv2.VideoCapture(vid10)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)

#Reading circle gesture class
listing11 = os.listdir('/home/ubuntu/dataset_rgb/circle')
for vid11 in listing11:
    vid11 = '/home/ubuntu/dataset_rgb/circle/'+vid11
    frames = []
    cap = cv2.VideoCapture(vid11)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)


    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[19]:

#Reading comehere gesture class

listing12 = os.listdir('/home/ubuntu/dataset_rgb/comehere')
for vid12 in listing12:
    vid12 = '/home/ubuntu/dataset_rgb/comehere/'+vid12
    frames = []
    cap = cv2.VideoCapture(vid12)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[20]:

#Reading cross gesture class

listing13 = os.listdir('/home/ubuntu/dataset_rgb/cross')
for vid13 in listing13:
    vid13 = '/home/ubuntu/dataset_rgb/cross/'+vid13
    frames = []
    cap = cv2.VideoCapture(vid13)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[21]:

#Reading pat gesture class

listing14 = os.listdir('/home/ubuntu/dataset_rgb/pat')
for vid14 in listing14:
    vid14 = '/home/ubuntu/dataset_rgb/pat/'+vid14
    frames = []
    cap = cv2.VideoCapture(vid14)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[22]:

#Reading rightleft gesture class

listing15 = os.listdir('/home/ubuntu/dataset_rgb/rightleft')
for vid15 in listing15:
    vid15 = '/home/ubuntu/dataset_rgb/rightleft/'+vid15
    frames = []
    cap = cv2.VideoCapture(vid15)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)



# In[23]:

#Reading triangle gesture class

listing16 = os.listdir('/home/ubuntu/dataset_rgb/triangle')
for vid16 in listing16:
    vid16 = '/home/ubuntu/dataset_rgb/triangle/'+vid16
    frames = []
    cap = cv2.VideoCapture(vid16)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[24]:

#Reading turaround gesture class

listing17 = os.listdir('/home/ubuntu/dataset_rgb/turnaround')
for vid17 in listing17:
    vid17 = '/home/ubuntu/dataset_rgb/turnaround/'+vid17
    frames = []
    cap = cv2.VideoCapture(vid17)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[25]:

#Reading updown gesture class

listing18 = os.listdir('/home/ubuntu/dataset_rgb/updown')
for vid18 in listing18:
    vid18 = '/home/ubuntu/dataset_rgb/updown/'+vid18
    frames = []
    cap = cv2.VideoCapture(vid18)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[26]:

#Reading wave gesture class

listing19 = os.listdir('/home/ubuntu/dataset_rgb/wave')
for vid19 in listing19:
    vid19 = '/home/ubuntu/dataset_rgb/wave/'+vid19
    frames = []
    cap = cv2.VideoCapture(vid19)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[27]:

#Reading z gesture class

listing20 = os.listdir('/home/ubuntu/dataset_rgb/z')
for vid20 in listing20:
    vid20 = '/home/ubuntu/dataset_rgb/z/'+vid20
    frames = []
    cap = cv2.VideoCapture(vid20)
    fps = cap.get(5)
    print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)

    for k in xrange(16):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    print input.shape
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    print ipt.shape

    X_tr.append(ipt)


# In[28]:

X_tr_array = np.array(X_tr)   # convert the frames read into array

num_samples = len(X_tr_array)
print num_samples

#Assign Label to each class
label=np.ones((num_samples,),dtype = int)
label[0:108]= 0
label[108:215] = 1
label[215:323] = 2
label[323:431] = 3
label[431:539]= 4
label[539:647] = 5
label[647:755] = 6
label[755:863] = 7
label[863:971] = 8
label[971:1079] = 9
label[1079:1187]= 0
label[1187:1295] = 1
label[1295:1403] = 2
label[1403:1511] = 3
label[1511:1619]= 4
label[1619:1727] = 5
label[1727:1835] = 6
label[1835:1943] = 7
label[1943:2051] = 8
label[2051:] = 9


train_data = [X_tr_array,label]
(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)
train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))
for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]

patch_size = 16    # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')

# CNN Training parameters
batch_size = 20
nb_classes = 10
nb_epoch =200

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
print len(Y_train)
# number of convolutional filters to use at each layer
#nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
#nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
#nb_conv = [5,5]

# Pre-processing
train_set = train_set.astype('float64')
train_set -= np.mean(train_set)
train_set /=np.max(train_set)


# In[15]:

# Define model
model = Sequential()
model.add(Convolution3D(8,nb_depth=2, nb_row=2, nb_col=2,border_mode='full',input_shape=(1,16,16,16),activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Convolution3D(16,nb_depth=2, nb_row=2, nb_col=2,border_mode='full',activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

#model.add(Convolution3D(64,nb_depth=3, nb_row=3, nb_col=3,border_mode='full', activation='relu'))
#model.add(MaxPooling3D(pool_size=(2,2,2)))
#model.add(Dropout(0.5))

#model.add(Convolution3D(128,nb_depth=3, nb_row=3, nb_col=3,border_mode='full', activation='relu'))
#model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())

#model.add(Dense(512, init='normal', activation='relu'))
#model.add(Dropout(0.20))

model.add(Dense(50, init='normal', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes,init='normal'))
model.add(Activation('softmax'))

rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer= rms)
#model.compile(loss='categorical_crossentropy',
#              optimizer='adadelta')


# In[16]:

# Split the data

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.1, random_state=5)


# Train the model

#hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
#          batch_size=batch_size,nb_epoch = nb_epoch,show_accuracy=True,shuffle=True)


hist = model.fit(X_train_new, y_train_new, batch_size=20,
         nb_epoch=200,validation_split=0.1, show_accuracy=True,
           shuffle=True)


 # Evaluate the model
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

# Plot the results
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(200)
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
#plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss')
plt.grid(True)
plt.legend(['train'])
plt.savefig('testplot10_1.png')
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
#plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc')
plt.grid(True)
plt.legend(['train'],loc=4)
plt.savefig('testplot10_2.png')
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])


# In[ ]:

#saving weights
fname = "weights-Test-CNN10.hdf5"
model.save_weights(fname,overwrite=True)
#model.save('depth_model.h5')
print(model.summary())
