import  numpy as np
import cv2
import sys
import time
import logging
import queue
from os import system
import os
import pickle
import threading
from threading import Thread
from datetime import datetime
import serial
import config
import flycapture2 as fc2
from tracker import track

# Multithreading Setup
logging.basicConfig(level=logging.DEBUG,
					format='(%(threadName)-9s) %(message)s',)
BUF_SIZE = 100
q = queue.Queue(BUF_SIZE)

def GenericCamera(n_frames, save_dir):
	cap = fc2.Context()
	cap.connect(*cap.get_camera_from_index(0))
	cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_15)
	m, f = cap.get_video_mode_and_frame_rate()
	p = cap.get_property(fc2.FRAME_RATE)
	cap.set_property(**p)
	cap.start_capture()

	photos = []

	for i in range(n_frames):

		img = fc2.Image()
		cap.retrieve_buffer(img)
		frame = np.array(img)
		photos.append(frame)
		# smallFrame = cv2.resize(frame, None, fx=0.5, fy=0.5)
		cv2.imshow("Frame", frame)
		#cv2.imwrite(save_dir+'/{}.jpg'.format(i), frame)

		# Video keyboard interrupt
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if save_dir is not None:
		with open(save_dir+'/photos.pkl', 'wb') as f:
			pickle.dump(photos, f)
	else:
		pass

	cap.stop_capture()
	cap.disconnect()
	cv2.destroyAllWindows()
	cv2.waitKey()

class ProducerCamera(Thread):
	def __init__(self, nFrames, saveDir, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		super(ProducerCamera,self).__init__()
		self.target = target
		self.name = name
		self.nFrames = nFrames
		self.save_dir = saveDir

	def run(self):
		cap = fc2.Context()
		cap.connect(*cap.get_camera_from_index(0))
		cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_15)
		m, f = cap.get_video_mode_and_frame_rate()
		p = cap.get_property(fc2.FRAME_RATE)
		cap.set_property(**p)
		cap.start_capture()

		photos = []
		slidingWindow = []

		for i in range(self.nFrames):
			img = fc2.Image()
			cap.retrieve_buffer(img)
			frame = np.array(img)

			photos.append(frame)
			detect, trackedFrame, pos = track(frame, 90, config.flySize)
			if pos[1] is None:
				slidingWindow.append(np.mean(slidingWindow))
			else:
				slidingWindow.append(pos[1])
			cv2.imshow("Frame", trackedFrame)
			#cv2.imwrite(self.saveDir+'/{}.jpg'.format(i), frame)

			# Code for walking direction
			# 0 = No walking
			# 1 = Walking towards top
			# 2 = Walking towards bottom
			targetYD1 = 0
			targetYD2 = 480

			if i > 15:
				slidingWindow.pop(0)
				d = np.diff(slidingWindow)
				da = np.mean(d)
				if da > 1:
					signal = 2
				elif da < -1:
					signal = 1
				else:
					signal = 0
			else:
				signal = 0

			if not q.full():
				# This will be set conditionally by the tracker
				item = signal
				q.put(item)
				#logging.debug('Putting {} in queue'.format(signal))

			# Video keyboard interrupt
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if self.save_dir is not None:
			with open(self.save_dir+'/photos.pkl', 'wb') as f:
				pickle.dump(photos, f)
		else:
			pass

		cap.stop_capture()
		cap.disconnect()
		cv2.destroyAllWindows()
		cv2.waitKey()

		return

def GenericSerial(ser, programLists, wrap):

	timeList = programLists[1]
	cfgList = programLists[0]

	ts=0
	for j in timeList:
		ts = ts+int(j)

	t0 = time.time()

	for index in range(len(cfgList)):

		print(int(100*((time.time() - t0)/ts)), '% done with this block')

		c = str(cfgList[index])

		ser.write(str.encode(c))
		#ser.write(cfgList[index])
		time.sleep(timeList[index])

	if wrap==True:
		ser.write(str.encode('0'))
		print('100 % done with this block')
	else:
		pass

class ConsumerSerial(Thread):
	def __init__(self,ser, programLists, wrap, group=None, target=None, name=None,
				 args=(), kwargs=None, verbose=None):
		super(ConsumerSerial,self).__init__()
		self.target = target
		self.name = name
		self.look = True
		self.ser = ser
		self.programLists = programLists
		self.wrap = wrap
		self.altCfgList = altCfgList
		return

	def checkForActivation(self):
		self.code = 0

		if not q.empty():
			item = q.get()
			if item==1:
				self.code=1
			elif item==2:
				self.code=2
			else:
				self.code=0
			#logging.debug('Getting ' + str(item))

	def run(self):
		timeList = self.programLists[0]
		cfgList = self.programLists[1]
		altCfgList = self.programLists[2]

		ts=0

		for j in timeList:
			ts = ts+int(j)
		tog = time.time()

		for index in range(len(cfgList)):
			t0 = time.time()

			print(int(100*((time.time() - tog)/ts)), '% done with this block')

			while (time.time() - t0) < timeList[index]:

				time.sleep(1/30)
				self.checkForActivation()

				# REVERSE LOGIC HERE IF YOU HAVE TO DO IT
				if self.code==1 and int(cfgList[index])==1:
					c = str(altCfgList[index])
				elif self.code==2 and int(cfgList[index]==2):
					c = str(altCfgList[index])
				else:
					c = str(cfgList[index])

				ser.write(str.encode(c))

		self.look=False

		if self.wrap==True:
			ser.write(str.encode('0'))
			print('100 % done with this block')
		else:
			pass

		return None

def getProgramLists():
	# Set up program for Block 1 and Block 3
	if config.lightColor=='green' and config.lightLevel=='low':
		cfgList = ['6', '1', '7','2']
	if config.lightColor=='green' and config.lightLevel=='high':
		cfgList = ['b', '1', 'c','2']
	if config.lightColor=='red' and config.lightLevel=='low':
		cfgList = ['g', '1', 'h','2']
	if config.lightColor=='red' and config.lightLevel=='high':
		cfgList = ['l', '1', 'm','2']

	timeList = [config.lightDur, (config.switchDur-config.lightDur), config.lightDur, (config.switchDur-config.lightDur)]

	preCfgList = cfgList
	preTimeList = timeList
	for i in range(config.nFullSwitchPre-1):
		preCfgList = preCfgList + cfgList
		preTimeList = preTimeList + timeList
	preProgramLists = [preCfgList, preTimeList]

	postCfgList = cfgList
	postTimeList = timeList
	for i in range(config.nFullSwitchPost - 1):
		postCfgList = postCfgList + cfgList
		postTimeList = postTimeList + timeList
	postProgramLists = [postTimeList, postTimeList]

	return preProgramLists, postProgramLists

def fileOps():
	dt = datetime.now()
	datetimeString = str(dt.month)+"_"+str(dt.day)+"_"+str(dt.year)+"_"+str(dt.hour)+str(dt.minute)
	prePath = config.saveDir + r"\{}\prereinforcement".format(datetimeString)
	rePath = config.saveDir + r"\{}\reinforcement".format(datetimeString)
	postPath = config.saveDir + r"\{}\postreinforcement".format(datetimeString)
	baseDir = config.saveDir + r"\{}".format(datetimeString)

	os.mkdir(baseDir)
	os.mkdir(prePath)
	os.mkdir(rePath)
	os.mkdir(postPath)

	return baseDir, prePath, rePath, postPath

if __name__=="__main__":
	baseDir, prePath, rePath, postPath = fileOps()
	preProgramLists, postProgramLists = getProgramLists()

	# # Pre Reinforcement
	# print('Pre-Reinforcement Block')
	# saveDir = prePath
	# nFrames = int(15*np.sum(preProgramLists[1]))
	# wrap = True
	# ser = serial.Serial(config.COMM, config.baud)
	# cameraThread = threading.Thread(target = GenericCamera, args=(nFrames, saveDir,))
	# serialThread = threading.Thread(target=GenericSerial, args=(ser, preProgramLists, wrap,))
	# cameraThread.start()
	# serialThread.start()
	# while (serialThread.isAlive() or cameraThread.isAlive()):
	# 	time.sleep(0.1)
	# ser.close()

	# Reinforcement
	print('Reinforcement Block')
	saveDir = rePath
	nFrames = int(15*2*config.switchDur*config.nFullSwitchReinforce)

	wrap = True

	timeList= [config.switchDur, config.switchDur]
	cfgList = ['1', '2']
	if config.lightColor=='green' and config.lightLevel=='low':
		altCfgList = ['6', '7']
	if config.lightColor=='green' and config.lightLevel=='high':
		altCfgList = ['b', 'c']
	if config.lightColor=='red' and config.lightLevel=='low':
		altCfgList = ['g', 'h']
	if config.lightColor=='red' and config.lightLevel=='high':
		altCfgList = ['l', 'm']

	reTimeList = timeList
	reCfgList = cfgList
	reAltCfgList = altCfgList
	for i in range(config.nFullSwitchPre-1):
		reTimeList = reTimeList + timeList
		reCfgList = reCfgList + cfgList
		reAltCfgList = reAltCfgList + altCfgList

	reProgramLists = [reTimeList, reCfgList, reAltCfgList]
	ser = serial.Serial(config.COMM, config.baud)
	pcam = ProducerCamera(nFrames = nFrames, saveDir = saveDir, name='producer')
	cser = ConsumerSerial(ser, reProgramLists, wrap, name='consumer')
	pcam.start()
	cser.start()

	while (pcam.isAlive() or cser.isAlive()):
		time.sleep(0.1)
	ser.close()
	print('done')

	# # Post Reinforcement
	# print('\nPost-Reinforcement Block')
	# saveDir = postPath
	# nFrames2 = int(15*np.sum(postProgramLists[1]))
	# wrap = True
	# ser = serial.Serial(config.COMM, config.baud)
	# cameraThread2 = threading.Thread(target = GenericCamera, args=(nFrames, saveDir,))
	# serialThread2 = threading.Thread(target=GenericSerial, args=(ser, postProgramLists, wrap,))
	# cameraThread2.start()
	# serialThread2.start()
	# while (serialThread.isAlive() or cameraThread.isAlive()):
	# 	time.sleep(0.1)
