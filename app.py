from PyQt5 import QtWidgets, uic,QtGui
import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import base64
import numpy as np
# import tensorflow_hub as hub
import argparse
import tensorflow as tf
import cv2
from PIL.ImageQt import ImageQt
from PIL import Image
print(cv2.__version__)
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
utils_ops.tf = tf.compat.v1
import threading
from ASL.src.models import create_model
from ASL.src.models.tresnet import TResnetM, TResnetL, TResnetXL
from ast import literal_eval
from PIL import Image
from ASL.src.models import create_model
import resources
import torch
from collections import Counter
import logo_rc


CONFIDENCE_SCORE = 0.90
	# Patch the location of gfile
tf.gfile = tf.io.gfile

class NetworkUi(QtWidgets.QMainWindow):
	changePixmap = pyqtSignal(QImage)

	def __init__(self):
		super(NetworkUi,self).__init__() #Call the inherited classes __init__ method
		uic.loadUi('test.ui', self)
		
		# self.Drone_camera.clicked.connect(self.runSlot)
		# self.Quadruped_camera.clicked.connect(self.runSlot2)
		# self.comboBox.activated.connect(self.runSlot)
		# self.comboBox_2.activated.connect(self.runSlot2)
		self.stop.clicked.connect(self.stop_feed)
		self.stop_3.clicked.connect(self.stop_feed2)
		self.comboBox.activated.connect(self.current_text)
		self.comboBox_2.activated.connect(self.current_text_2)
	

	def current_text(self, _): # We receive the index, but don't use it.
		self.ctext = self.comboBox.currentText()
		print("Current text", self.ctext)
		if self.ctext=="Object Detection Count":
			self.runSlot()
		elif self.ctext=="MultiLabel Classification":
			self.runSlot2()

	def current_text_2(self, _): # We receive the index, but don't use it.
		self.ctext = self.comboBox_2.currentText()
		print("Current text", self.ctext)
		if self.ctext=="Object Detection Count":
			self.runSlot_ob()
		elif self.ctext=="MultiLabel Classification":
			self.runSlot2_ml()

	def load_model(self,model_path):
		model = tf.saved_model.load(model_path)
		print(list(model.signatures.keys())) 
		return model

	def load_model_ASL(self,args):
		
		print('creating and loading the model...')
		state = torch.load(args['model_path'], map_location='cpu')
		num_classes = state['num_classes']
		print(num_classes)
		args['num_classes'] = num_classes

	# args.num_classes = state['num_classes']
		model = create_model(args).cuda()
		model.load_state_dict(state['model'], strict=True)
		model.eval()
		classes_list = np.array(list(state['idx_to_class'].values()))
		print('done\n')
		return model,classes_list

	@pyqtSlot()
	def stop_feed(self):
		self.cap.release()
		print("feed was asked to stop")

	@pyqtSlot()
	def stop_feed2(self):
		self.cap.release()
		print("feed was asked to stop")	

	def run_inference_for_image(self,args,model,classes_list, tensor_batch):
		print('loading image and doing inference...')    
		output, inter_op = model(tensor_batch)
		output = torch.squeeze(torch.sigmoid(output))
		np_output = list(output.cpu().detach().numpy())
		indices = [index for index, val in enumerate(np_output) if val > 0.95]
		probs = [float(np_output[index]) for index in indices]
		# np_output=np_output>0.5
		detected_classes = [classes_list[index] for index in indices]
		
		# print(detected_classes)
		assert len(probs) == len(detected_classes)
		out={literal_eval(detected_classes[index]):probs[index] for index in range(len(probs))}
		return out


	def run_inference_new(self,args,model, classes_list, cap):#out
		while True:
			ret, image_np = cap.read()
			
			im_resize = cv2.resize(image_np,(args['input_size'], args['input_size']))

			np_img = np.array(im_resize, dtype=np.uint8)
			tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
			tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
			# Actual detection.q
		
			output_dict = self.run_inference_for_image(args,model,classes_list, tensor_batch)#out
			window_name = 'Image'
	
			# font
			font = cv2.FONT_HERSHEY_SIMPLEX
	
			# org
			org = (50, 50)
	
			# fontScale
			fontScale = 1
	
			# Blue color in BGR
			color = (255, 0, 0)
	
			# Line thickness of 2 px
			thickness = 2
			# Using cv2.putText() method
			list1=output_dict.keys()
			text=(",".join(list1))
			print(text)
			text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
			line_height = text_size[1] + 5
			print(line_height)
			# image_np = cv2.rectangle(image_np, (0, 0), (1280, 50), (0,125,255), -1)

			y0, dy = 30, 4
			for i, line in enumerate(list1):
				y = y0 + i * line_height
				# y = y0 + i*dy
				# cv2.putText(image_np, line, (30, y ), cv2.FONT_HERSHEY_SIMPLEX,1,color, 2)
				# line = "avinash \n singh"
				cv2.putText(image_np, line, (30, y ),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
				image_np = cv2.resize(image_np, (960, 540)) 

			frame = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
			ConvertToQtFormat = QImage(frame, image_np.shape[1], image_np.shape[0], QImage.Format_RGB888)
			self.Screen1.setPixmap(QPixmap.fromImage(ConvertToQtFormat))
			self.Screen1.setScaledContents(True)
			# image = cv2.putText(image_np, item, org, font, cd
			#         fontScale, color, thickness, cv2.LINE_AA)
				
			#  Displaying the image
			# cv2.imshow(window_name, image_np)
			# # out.write(image_np)

			# if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	# out.release()
			# 	cap.release()
				
			# 	cv2.destroyAllWindows()
			# 	break 
	def run_inference_ml(self,args,model, classes_list, cap):#out
		while True:
			ret, image_np = cap.read()
			
			im_resize = cv2.resize(image_np,(args['input_size'], args['input_size']))

			np_img = np.array(im_resize, dtype=np.uint8)
			tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
			tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
			# Actual detection.q
		
			output_dict = self.run_inference_for_image(args,model,classes_list, tensor_batch)#out
			window_name = 'Image'
	
			# font
			font = cv2.FONT_HERSHEY_SIMPLEX
	
			# org
			org = (50, 50)
	
			# fontScale
			fontScale = 1
	
			# Blue color in BGR
			color = (255, 0, 0)
	
			# Line thickness of 2 px
			thickness = 2
			# Using cv2.putText() method
			list1=output_dict.keys()
			text=(",".join(list1))
			print(text)
			text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
			line_height = text_size[1] + 5
			print(line_height)
			# image_np = cv2.rectangle(image_np, (0, 0), (831, 50), (0,125,255), -1)

			y0, dy = 30, 4
			for i, line in enumerate(list1):
				y = y0 + i * line_height
				# y = y0 + i*dy
				# cv2.putText(image_np, line, (30, y ), cv2.FONT_HERSHEY_SIMPLEX,1,color, 2)
				cv2.putText(image_np, line, (30, y ),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
				image_np = cv2.resize(image_np, (960, 540)) 

			frame = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
			ConvertToQtFormat = QImage(frame, image_np.shape[1], image_np.shape[0], QImage.Format_RGB888)
			self.Screen2.setPixmap(QPixmap.fromImage(ConvertToQtFormat))
			self.Screen2.setScaledContents(True)


	def run_inference_for_single_image(self,model, image):
		image = np.asarray(image)
		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(image)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis,...]
		
		# Run inference
		output_dict = model(input_tensor)

		# All outputs are batches tensors.
		# Convert to numpy arrays, and take index [0] to remove the batch dimension.
		# We're only interested in the first num_detections.
		num_detections = int(output_dict.pop('num_detections'))
		output_dict = {key: value[0, :num_detections].numpy()
					for key, value in output_dict.items()}
		output_dict['num_detections'] = num_detections

		# detection_classes should be ints.
		output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
	
		# Handle models with masks:
		if 'detection_masks' in output_dict:
			# Reframe the the bbox mask to the image size.
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
										output_dict['detection_masks'], output_dict['detection_boxes'],
										image.shape[0], image.shape[1])      
			detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
			output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
		
		return output_dict


	def filter_predictions(self,pred_dict:dict, threshold:int):


		detection_boxes = pred_dict['detection_boxes'].tolist()
		detection_scores = pred_dict['detection_scores'].tolist()
		class_names = pred_dict['detection_classes'].tolist()

		# filterd_detection_classes_indices = [index for index, class_name in enumerate(class_names) if len(class_names) >=3]
		filterd_detection_scores_indices = [index for index, score in enumerate(detection_scores) if score >= threshold]
		pred_dict['detection_boxes'] = [detection_boxes[index] for index in filterd_detection_scores_indices]
		pred_dict['detection_classes'] = [class_names[index] for index in filterd_detection_scores_indices]
		pred_dict['detection_scores'] = [detection_scores[index] for index in filterd_detection_scores_indices]
		return pred_dict


	def run_inference(self,model, category_index, cap):
		while True:
			ids=[]
			names=[]
			class_names=[]
			count_dict={'class_names':[]}
			ret, image_np = cap.read()
			# Actual detection.q
			output_dict = self.run_inference_for_single_image(model, image_np)
			# Visualization of the results of a detection.
			output_dict=self.filter_predictions(output_dict,.50)
			print(len(output_dict))

			for key in category_index:
					for value in category_index[key]:
						# print("the dict of dict id is :", i)
						if value=='id':
							ids.append(category_index[key][value])
						# print(ids)
						if value=='name':
							names.append(category_index[key][value])
						# print(name)

			classes=output_dict['detection_classes']
			dictionary = dict(zip(ids, names))    
			# print(dictionary) 

			for item in classes:
				class_names.append(dictionary[item])
			
			count_dict['class_names']=class_names
			
			
			
			
			count=dict(Counter(count_dict['class_names']))
			# print(count)
			l=[]
			for key,value in count.items():
				l.append(key+ '('+str(value)+')')
			output_dict['summary']=', '.join(l)

			print(output_dict['summary'])
			
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.array(output_dict['detection_boxes']),
				np.array(output_dict['detection_classes']),
				np.array(output_dict['detection_scores']),
				category_index,
				instance_masks=output_dict.get('detection_masks_reframed', None),
				use_normalized_coordinates=True,
				line_thickness=8)

			frame = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
			cv2.putText(frame, output_dict['summary'], (10, 20 ),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
			ConvertToQtFormat = QImage(frame, image_np.shape[1], image_np.shape[0], QImage.Format_RGB888)
			# Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
			# self.changePixmap.emit(Pic)

			
			self.Screen1.setPixmap(QPixmap.fromImage(ConvertToQtFormat))
			self.Screen1.setScaledContents(True)
			
			

		

			# cv2.imshow('object_detection',image_np)#cv2.resize(image_np, (800, 600))
			# if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	cap.release()
			# 	cv2.destroyAllWindows()
			# 	break

	def run_inference_ob(self,model, category_index, cap):
		# ids=[]
		# names=[]
		# class_names=[]
		# count_dict={'class_names':[]}
		while True:
			ids=[]
			names=[]
			class_names=[]
			count_dict={'class_names':[]}
			
			ret, image_np = cap.read()
			# Actual detection.q
			output_dict = self.run_inference_for_single_image(model, image_np)
			# print(type(out_dict['detection_boxes']))
			
			output_dict=self.filter_predictions(output_dict,.50)
			print(len(output_dict))

			for key in category_index:
					for value in category_index[key]:
						# print("the dict of dict id is :", i)
						if value=='id':
							ids.append(category_index[key][value])
						# print(ids)
						if value=='name':
							names.append(category_index[key][value])
						# print(name)

			classes=output_dict['detection_classes']
			dictionary = dict(zip(ids, names))    
			# print(dictionary) 

			for item in classes:
				class_names.append(dictionary[item])
			
			count_dict['class_names']=class_names
			
			
			
			
			count=dict(Counter(count_dict['class_names']))
			# print(count)
			l=[]
			for key,value in count.items():
				l.append(key+ '('+str(value)+')')
			output_dict['summary']=', '.join(l)

			print(output_dict['summary'])

			# Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.array(output_dict['detection_boxes']),
				np.array(output_dict['detection_classes']),
				np.array(output_dict['detection_scores']),
				category_index,
				instance_masks=output_dict.get('detection_masks_reframed', None),
				use_normalized_coordinates=True,
				line_thickness=8)

			frame = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
			cv2.putText(frame, output_dict['summary'], (10, 20 ),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
			ConvertToQtFormat = QImage(frame, image_np.shape[1], image_np.shape[0], QImage.Format_RGB888)
			# Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
			# self.changePixmap.emit(Pic)

			
			self.Screen2.setPixmap(QPixmap.fromImage(ConvertToQtFormat))
			self.Screen2.setScaledContents(True)
	

	# @pyqtSlot()
	def runSlot(self):
		
		print("inside drone")
		model="./ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/"
		labelmap= "./label_map_600/mscoco_label_map.pbtxt"

		detection_model = self.load_model(model)
		category_index = label_map_util.create_category_index_from_labelmap(labelmap, use_display_name=True)

		self.cap = cv2.VideoCapture(0)#("rtsp://192.168.43.1:8554/fpv_stream",cv2.CAP_FFMPEG)
		# cap.set(3,640) #set frame width
		# cap.set(4,480) #set frame height
		# cap.set(cv2.CAP_PROP_FPS, 15) #adjusting fps to 5
		# self.run_inference(detection_model, category_index, cap)
		thread1 = threading.Thread(target=self.run_inference, 
                           args=( detection_model, category_index, self.cap))
		
		thread1.start()

		
	# @pyqtSlot()
	def runSlot2(self):
		print("Clicked Run")
		args = {
		'model_path': './ASL/weights/Open_ImagesV6_TRresNet_L_448.pth',
		'model_name': 'tresnet_l',
		'input_size':448,
		'dataset_type' : 'OpenImages',
		'th':CONFIDENCE_SCORE,
		'do_bottleneck_head' :True

		}
		detection_model,classes_list = self.load_model_ASL(args)
		self.cap = cv2.VideoCapture(0)#("rtsp://192.168.43.1:8554/fpv_stream",cv2.CAP_FFMPEG
    # cap.set(3,640) #set frame width
    # cap.set(4,480) #set frame height
	# cap.set(cv2.CAP_PROP_FPS, 1)
		thread2 = threading.Thread(target=self.run_inference_new ,
                           args=( args,detection_model,classes_list, self.cap))
		
		thread2.start()

		# detect_from_webcam.main_code()
	



	def runSlot_ob(self):
		
		print("inside drone")
		model="./ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/"
		labelmap= "./label_map_600/mscoco_label_map.pbtxt"

		detection_model = self.load_model(model)
		category_index = label_map_util.create_category_index_from_labelmap(labelmap, use_display_name=True)

		self.cap = cv2.VideoCapture(0)#("rtsp://192.168.43.1:8554/fpv_stream",cv2.CAP_FFMPEG)
		# cap.set(3,640) #set frame width
		# cap.set(4,480) #set frame height
		# cap.set(cv2.CAP_PROP_FPS, 15) #adjusting fps to 5
		# self.run_inference(detection_model, category_index, cap)
		thread1 = threading.Thread(target=self.run_inference_ob, 
                           args=( detection_model, category_index, self.cap))
		
		thread1.start()

		
	# @pyqtSlot()
	def runSlot2_ml(self):
		print("Clicked Run")
		args = {
		'model_path': './ASL/weights/Open_ImagesV6_TRresNet_L_448.pth',
		'model_name': 'tresnet_l',
		'input_size':448,
		'dataset_type' : 'OpenImages',
		'th':CONFIDENCE_SCORE,
		'do_bottleneck_head' :True

		}
		detection_model,classes_list = self.load_model_ASL(args)
		self.cap = cv2.VideoCapture(0)#("rtsp://192.168.43.1:8554/fpv_stream",cv2.CAP_FFMPEG
    # cap.set(3,640) #set frame width
    # cap.set(4,480) #set frame height
	# cap.set(cv2.CAP_PROP_FPS, 1)
		thread2 = threading.Thread(target=self.run_inference_ml ,
                           args=( args,detection_model,classes_list, self.cap))
		
		thread2.start()	

if __name__ == "__main__":		
	app = QtWidgets.QApplication(sys.argv)
	window = NetworkUi()
	window.show()
	app.exec_()