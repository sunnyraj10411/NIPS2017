"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf

from inception_resnet_v2 import *
import cv2
import csv

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 20, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images1 = np.zeros(batch_shape)
  images2 = np.zeros(batch_shape)
  images3 = np.zeros(batch_shape)
  images4 = np.zeros(batch_shape)

  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
        for manType in range(4):
            img = cv2.imread(filepath)
            #flip and denoise
            if manType == 0:
                dst = cv2.fastNlMeansDenoisingColored(img,None,7,10,7,21)
                dst = cv2.flip(dst,1)
                cv2.imwrite('./store/'+'temp.png',dst)
                image = imread('./store/'+'temp.png', mode='RGB').astype(np.float) / 255.0
                images1[idx, :, :, :] = image * 2.0 - 1.0
            #denoise no flip
            elif manType == 1:
                dst = cv2.fastNlMeansDenoisingColored(img,None,7,10,7,21)
                #dst = cv2.flip(dst,1)
                cv2.imwrite('./store/'+'temp.png',dst)
                image = imread('./store/'+'temp.png', mode='RGB').astype(np.float) / 255.0
                images2[idx, :, :, :] = image * 2.0 - 1.0
            #only flip
            elif manType == 2:   
                #dst = cv2.fastNlMeansDenoisingColored(img,None,7,10,7,21)
                dst = cv2.flip(img,1)
                cv2.imwrite('./store/'+'temp.png',dst)
                image = imread('./store/'+'temp.png', mode='RGB').astype(np.float) / 255.0
                images3[idx, :, :, :] = image * 2.0 - 1.0
            else:
                cv2.imwrite('./store/'+'temp.png',img)
                image = imread('./store/'+'temp.png', mode='RGB').astype(np.float) / 255.0
                # Images for inception classifier are normalized to be in [-1, 1] interval.
                images4[idx, :, :, :] = image * 2.0 - 1.0
    
        filenames.append(os.path.basename(filepath))
        idx += 1
    if idx == batch_size:
        yield filenames, images1,images2,images3,images4
        filenames = []
        images1 = np.zeros(batch_shape)
        images2 = np.zeros(batch_shape)
        images3 = np.zeros(batch_shape)
        images4 = np.zeros(batch_shape)
        idx = 0
  if idx > 0:
    yield filenames, images1,images2,images3,images4

def findMax(labelList,valList,pref):

    #print(labelList)
    #print(valList)
    check = {}
    #print(labelList)
    #exit()
    count = 0
    
    for i in range(len(labelList)):
        for j in range(len(labelList[0])):
            check[labelList[i][j]] = 0

    for i in range(len(labelList)):
        for j in range(len(labelList[0])):
            check[labelList[i][j]] += valList[i][j]
   
    v = list(check.values())
    k = list(check.keys())
    maxIndex = k[v.index(max(v))]
    maxVal = max(v)
    v[v.index(max(v))] = 0
    maxVal1 = max(v)
    maxIndex1 = k[v.index(max(v))]
    #print(check)
    return maxIndex,maxIndex1,maxVal,maxVal1



def satya(sess,image,x_input,source,target,maxVal1,maxVal2,logits,itr):
    print("Entering satya")
    topPredictT = tf.nn.top_k(logits,k=1,sorted=True,name=None)
    topTwoPredictT = tf.nn.top_k(logits,k=5,sorted=True,name=None)
    topPredict = sess.run(topPredictT,feed_dict={x_input:image})
    topTwoPredict = sess.run(topTwoPredictT,feed_dict={x_input:image})
    print(topTwoPredictT)

    print(topTwoPredict[1][0][1])

    chImage = image
    labels = np.zeros(FLAGS.batch_size)
    for i in range(FLAGS.batch_size):
        if maxVal1[i]-maxVal2[i] > 3.4:
            print("Not doing anything for ",i," - ",maxVal1[i]-maxVal2[i])
            labels[i] = source[i]
            source[i] = 0
    #exit()
    
    j = 0
    allAdGen = 0

    err = sess.run(logits,feed_dict={x_input: chImage})
    gradientsErrT, = tf.gradients(logits,x_input,err)

    while j < itr and allAdGen == 0 :
        j = j+1
        preditop = sess.run(topTwoPredictT,feed_dict={x_input: chImage})
        print(preditop)
        err = sess.run(logits,feed_dict={x_input: chImage})

        maxOneStore = np.empty(FLAGS.batch_size)
        maxTwoStore = np.empty(FLAGS.batch_size)
        for i in range(FLAGS.batch_size):
            #pos = batchNo*FLAGS.batch_size+i
            #print(topPredict[1][i][0],"v",source[pos],"pos",pos,"len",len(source))
            #print("topPredict ",topPredict[1][i][0],"i",i,"size",len(source),"vs",source[i])
            if topPredict[1][i][0] != source[i]:
                labels[i] = topPredict[1][i][0]
                continue
            maxOneStore[i] = err[i,source[i]]
            maxTwoStore[i] = err[i,target[i]]
            #print("i",i,"source",maxOneStore[i],"destination",maxTwoStore[i])
            #cor[i,source[pos]] = 0
            for k in range(1001):
                if k == target[i]:
                    err[i,k] = 1
                else:
                    err[i,k] = -err[i,k]

        #gradientsErrT, = tf.gradients(logits,x_input,err)
        gradientsErr = sess.run(gradientsErrT,feed_dict={x_input:chImage})

        for i in range(FLAGS.batch_size):
            pert = np.inf
            if topPredict[1][i][0] != source[i]:
                    continue
            wK = gradientsErr[i] #- gradientsCor[i]
            fK = maxOneStore[i] - maxTwoStore[i] 
            pertK = (abs(fK)+0.02)/np.linalg.norm(wK.flatten())
            #print("fk ",abs(fK),"for index",i,"pertK",pertK)
            if pertK < pert:
                pert = pertK
                w = wK

            rI = (pert/10)*w #/np.linalg.norm(w)
            chImage[i,...] = chImage[i,...]+rI  
            #chImage[i,...] = chImage[i,...]+wK
        
        topPredict = sess.run(topPredictT,feed_dict={x_input: chImage })
        #print("topPredict ",topPredict)
        allAdGen = 1
        for i in range(FLAGS.batch_size):
            if topPredict[1][i][0] == source[i]:
                allAdGen = 0
                break
    
    for i in range(FLAGS.batch_size):
        if labels[i] == 0:
            labels[i] = source[i]
        if labels[i] == topTwoPredict[1][i][1]:
            labels[i] = source[i]
    return labels


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  img_shape = [FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  correctLabel = {}
  with open('dummy','r') as csvfile:
     reader = csv.reader(csvfile,delimiter=',')
     for row in reader:
         correctLabel[row[0]] = int(row[1])

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
      logits , end_points = inception_resnet_v2( x_input,is_training=False )

    #predicted_labels = tf.argmax(end_points['Predictions'], 1)
    predicted_labels = tf.nn.top_k(end_points['Predictions'],k=2,sorted=True,name=None)
    # Run computation
    saver = tf.train.Saver()

    manType = 1
    with tf.Session() as sess:
        saver.restore(sess,"./ens_adv_inception_resnet_v2.ckpt")
        with tf.gfile.Open(FLAGS.output_file,'w') as out_file:
            for filenames, images1,images2,images3,images4 in load_images(FLAGS.input_dir, batch_shape):
                labelsStore = []
                filenamesStore = []

                labelsStore.append(sess.run(predicted_labels, feed_dict={x_input: images1}))
                labelsStore.append(sess.run(predicted_labels, feed_dict={x_input: images2}))
                labelsStore.append(sess.run(predicted_labels, feed_dict={x_input: images3}))
                labelsStore.append(sess.run(predicted_labels, feed_dict={x_input: images4}))
                #print(labelsStore)
                #exit()
                print(len(filenames))
                if len(filenames) < FLAGS.batch_size:
                    for k in range(FLAGS.batch_size - len(filenames)):
                        filenames.append("dummy")
                filenamesStore = filenames
                #imageStore.append(images)
   
                #print(filenamesStore)

                sourceList = []
                targetList = []
                maxValList = []
                maxVal1List = []

                for image1 in range(FLAGS.batch_size):
                    labelList = []
                    valList = []
                    for i in range(4):
                        #print(image)#,dictStore[i][image])
                        labelList.append(labelsStore[i][1][image1])
                        valList.append(labelsStore[i][0][image1])
                    
                    #print(valList,labelList)
                    
                    index0,index1,maxVal,maxVal1 = findMax(labelList,valList,3)
                    print(image1,index0,index1,maxVal,maxVal1)
                    sourceList.append(index0)
                    targetList.append(index1)
                    maxValList.append(maxVal)
                    maxVal1List.append(maxVal1)

                #label = satya(sess,images1,x_input,sourceList,targetList,maxValList,maxVal1List,end_points["Predictions"],50)
                label = sourceList
                print(label)

                for j in range(len(label)):
                    if filenamesStore[j] != "dummy":# and correctLabel[filenamesStore[j]] != label[j] :
                        #print("Incorrect label for:",filenamesStore[j]," cl ",correctLabel[filenamesStore[j]]," wl ",label[j])
                        #print(dictStore[i][image])
                        out_file.write('{0},{1}\n'.format(filenamesStore[j],label[j]))
                    elif filenamesStore[j] != "dummy" :
                        #print("Correct label for:",filenamesStore[j]," cl ",correctLabel[filenamesStore[j]]," wl ",label[j])
                        #print(dictStore[i][image])
                        out_file.write('{0},{1}\n'.format(filenamesStore[j],int(label[j])))
                    else:
                        pass


if __name__ == '__main__':
  tf.app.run()

