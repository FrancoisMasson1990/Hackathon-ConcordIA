# Concordia Hackathon 11 nov
# Code made by Mathew, Zack and Francois

# Libraries and dependencies importation

import pandas as pd
import numpy as np
import sys, os
sys.path.append('align/detect_face.py')
sys.path.append('facenet.py')
sys.path.append('Sentiment.py')
import Sentiment as sent
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import subprocess as sub
from moviepy.editor import *
import align.detect_face as detect_face
import facenet
from PIL import Image
from yolo import YOLO
import time

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import warnings
warnings.filterwarnings('ignore')

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.65
fontColor              = (255,255,255)
lineType               = 2

def nets(sess, dir_):

	return detect_face.create_mtcnn(sess, 'align/')

# Webcam connection and image processing

def webcam_implementation(pnet, rnet, onet, sentiment_model, yolo):

	writeVideo_flag = True

	video_capture = cv2.VideoCapture(0)

	if writeVideo_flag:
		# Define the codec and create VideoWriter object
		w = int(video_capture.get(3))
		h = int(video_capture.get(4))
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out = cv2.VideoWriter('Webcam_Output/output.avi', fourcc, 15, (w, h))
		list_file = open('detection.txt', 'w')
		frame_index = -1 

	fps = 0.0
	while True:
		ret, frame = video_capture.read()  # frame shape 640*480*3
		t1 = time.time()
		if ret != True:
                    break

		crops, new_frames, crops_idc = process_video([frame], pnet, rnet, onet, sentiment_model)
		newer_frames,boxs = human_tracking(new_frames, yolo)
		print(newer_frames)
		cv2.imshow('', newer_frames[0])
		if writeVideo_flag:
			# save a frame
			out.write(newer_frames[0])
			frame_index = frame_index + 1
			list_file.write(str(frame_index)+' ')

			if len(boxs) != 0:
				for i in range(0,len(boxs)):
					list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
			list_file.write('\n')

		fps  = ( fps + (1./(time.time()-t1)) ) / 2
		print("fps= %f"%(fps))

		# Press Q to stop!
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	if writeVideo_flag:
		out.release()
		list_file.close()
	cv2.destroyAllWindows()

def human_tracking(frames, yolo):

	# Definition of the parameters
	max_cosine_distance = 0.3
	nn_budget = None
	nms_max_overlap = 1.0

	# deep_sort 
	model_filename = 'model_data/mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename,batch_size=1)

	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)

	new_frames = []

	for frame in frames:
		
		image = Image.fromarray(frame)
		boxs = yolo.detect_image(image)
		# print("box_num",len(boxs))
		features = encoder(frame,boxs)

		# score to 1.0 here).
		detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

		# Run non-maxima suppression.
		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]

		# Call the tracker
		tracker.predict()
		tracker.update(detections)

		for track in tracker.tracks:
		    if not track.is_confirmed() or track.time_since_update > 1:
		        continue 
		    bbox = track.to_tlbr()
		    frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
		    frame = cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

		for det in detections:
		    bbox = det.to_tlbr()
		    frame = cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

		new_frames.append(frame)

	return new_frames, boxs
	    

def process_video(frames, pnet, rnet, onet, sentiment_model):

	crops = []
	crop_idcs = []
	new_frames = []
	score = 100
	emotion = 'NA'

	for i, frame in enumerate(frames):

		bbox, _ = detect_face.detect_face(frame, minsize, 
		                                    pnet, rnet, onet, 
		                                    threshold, factor)

		frame = frame.copy()

		try:

			for box in bbox:

				w = box[2] - box[0]
				h = box[3] - box[1]
				#   plot the box using cv2
				crop_frame = frame[int(box[1]):int(box[1]+h), 
				               int(box[0]):int(box[0]+w)]

				if(i % 5 == 0):

					emotion, score = sent.Sentiment_Analysis(crop_frame, sentiment_model)

				frame = cv2.putText(frame,'%s %.3f%%' % (emotion, score), 
					    (int(box[0]), int(box[1]-5)), 
					    font, 
					    fontScale,
					    fontColor,
					    lineType)

				crops.append(crop_frame)
				crop_idcs.append(i)
				frame = cv2.rectangle(frame,(int(box[0]), int(box[1])), (int(box[0]+w), int(box[1]+h)),(0,0,255),2)

		except Exception as e:
			print(e)

		new_frames.append(frame)

	return crops, new_frames, crop_idcs

def get_embeddings(crops):

	with tf.Graph().as_default():

		with tf.Session() as sess:

			# Load the model
			facenet.load_model('models/')

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			images = [cv2.resize(crop, (160, 160)) for crop in crops]
			batches = [i for i in range(0, len(images), 32)] + [len(images)]
			embed_array = []
			print('BATCHES', batches)

			for i in range(1, len(batches)):
				print('BATCH', i)
				feed_dict = { images_placeholder: images[batches[i-1]:batches[i]], 
				phase_train_placeholder : False}
				# Use the facenet model to calcualte embeddings
				embed = sess.run(embeddings, feed_dict=feed_dict)
				embed_array.extend(embed.tolist())

			np.save('embeddings.npy', embed_array)

if __name__ == '__main__':

	sess = tf.Session()
	#Models
	pnet, rnet, onet = nets(sess, 'models/')
	sentiment_model = sent.Transfer_learning()
	yolo = YOLO()
	Video = 0
	if(Video == 1):
		print('Still')
		clip = VideoFileClip('Video/Video_1.mp4')
		fps = clip.fps
		print(fps)
		crops, new_frames, crop_idcs = process_video(clip.subclip(0, 10).iter_frames(), pnet, rnet, onet, sentiment_model)
		newer_frames = human_tracking(new_frames, yolo)
		np.save('face_crops.npy', crops)
		get_embeddings(crops)
		clip = ImageSequenceClip(new_frames, fps=fps)
		clip.write_videofile('Video_Output/newvideo.mp4', fps=fps)
	else:
		print('Webcam')
		webcam_implementation(pnet, rnet, onet, sentiment_model, yolo)

