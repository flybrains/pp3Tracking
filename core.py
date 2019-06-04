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
import csv

# Multithreading Setup

dt = datetime.now()
datetimeString = str(dt.month)+"_"+str(dt.day)+"_"+str(dt.year)+"_"+str(dt.hour)+str(dt.minute)
dts = config.saveDir + r"\{}".format(datetimeString)
os.mkdir(dts)
globalZeroTime = time.time()


logging.basicConfig(filename=dts+r'\log_raw.txt', level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
BUF_SIZE = 100
q = queue.Queue(BUF_SIZE)

def launch_FLIR_GUI(bg=None):

    if bg is None:
        cap = fc2.Context()
        cap.connect(*cap.get_camera_from_index(0))
        cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_15)
        m, f = cap.get_video_mode_and_frame_rate()
        p = cap.get_property(fc2.FRAME_RATE)
        cap.set_property(**p)
        cap.start_capture()
        img = fc2.Image()
        cap.retrieve_buffer(img)
        frame = np.array(img)
        
        im = frame.copy()
        im = np.expand_dims(im, 2)

        r = cv2.selectROI(im, fromCenter=False)

        cap.stop_capture()
        cap.disconnect()
        cv2.destroyAllWindows()

    else:
        r = cv2.selectROI(bg, fromCenter=False)
        cv2.destroyAllWindows()

    return r

def GenericCamera(n_frames, save_dir, block):
    cap = fc2.Context()
    cap.connect(*cap.get_camera_from_index(0))
    cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_15)
    m, f = cap.get_video_mode_and_frame_rate()
    p = cap.get_property(fc2.FRAME_RATE)
    cap.set_property(**p)
    cap.start_capture()

    photos = []

    time0 = time.time()
    for i in range(n_frames):
        print(time.time() - time0)

        if block=='pre':
            logging.debug("{},{},{}".format(str(time.time() - globalZeroTime), 'pre', str(i)))
        else:
            logging.debug("{},{},{}".format(str(time.time() - globalZeroTime), 'post', str(i)))

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
    def __init__(self, nFrames, saveDir, r, mask, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ProducerCamera,self).__init__()
        self.target = target
        self.name = name
        self.nFrames = nFrames
        self.save_dir = saveDir
        self.bg = bg

    def run(self):
        cap = fc2.Context()
        cap.connect(*cap.get_camera_from_index(0))
        cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_15)
        m, f = cap.get_video_mode_and_frame_rate()
        p = cap.get_property(fc2.FRAME_RATE)
        cap.set_property(**p)
        cap.start_capture()

        cy = r[1] + int(r[3]/2)
        cx = r[0] + int(r[2]/2)

        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]

        photos = []
        slidingWindow = []

        for i in range(self.nFrames):
            img = fc2.Image()
            cap.retrieve_buffer(img)
            frame = np.array(img)

            photos.append(frame)


            smallFrame = frame[y:y+h , x:x+w]
            smallFrame = np.multiply(smallFrame, mask)
            smallFrame[np.where(smallFrame == 0)] = 255


            detect, trackedFrame, pos = track(smallFrame, 120, config.flySize)
            
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
        

            if i > 2:
                slidingWindow.pop(0)
                d = np.diff(slidingWindow)
                da = np.mean(d)

                if da > 2:
                    signal = 2
                elif da < -2:
                    signal = 1
                else:
                    signal = 0
            else:
                signal = 0

            logging.debug('{},{},{},{},{}'.format(str(time.time()- globalZeroTime), 'reinforcement', str(i), str(pos[0]), str(pos[1]))), 

                
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

def GenericSerial(ser, programLists, wrap, block):

    timeList = programLists[1]
    cfgList = programLists[0]

    ts=0
    for j in timeList:
        ts = ts+int(j)

    t0 = time.time()

    for index in range(len(cfgList)):

        #print(int(100*((time.time() - t0)/ts)), '% done with this block')

        c = str(cfgList[index])

        if c in ['1','l','g','b','6']:
            direction = 'dir::DIR1'
        else:
            direction = 'dir::DIR2'

        if c in ['6', 'b', 'g', 'l', '7', 'c', 'h', 'm']:
            light = 'light::ON'
        else:
            light = 'light::OFF'


        if block=='pre':
            logging.debug("{},{},{},{}".format(str(time.time() - globalZeroTime), 'pre', direction, light))
        else:
            logging.debug("{},{},{},{}".format(str(time.time() - globalZeroTime), 'post', direction, light))


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
        #self.ser = ser
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

                time.sleep(0.001)
                self.checkForActivation()

                # REVERSE LOGIC HERE IF YOU HAVE TO DO IT
                if int(self.code)==1 and int(cfgList[index])==2:
                    c = str(altCfgList[index])
                elif int(self.code)==2 and int(cfgList[index])==1:
                    c = str(altCfgList[index])
                else:
                    c = str(cfgList[index])

                if c in ['1','l','g','b','6']:
                    direction = 'dir::DIR1'
                else:
                    direction = 'dir::DIR2'

                if c in ['6', 'b', 'g', 'l', '7', 'c', 'h', 'm']:
                    light = 'light::ON'
                else:
                    light = 'light::OFF'

                logging.debug('{},{},{},{}'.format(str(time.time() - globalZeroTime), 'reinforcement', direction, light))


                #print('Air Flow in Direction {}'.format(cfgList[index]))

                ser.write(str.encode(c))
                # print(c)

                

        self.look=False

        if self.wrap==True:
            ser.write(str.encode('0'))
            #print('100 % done with this block')
        else:
            pass

        return None

def getProgramLists():
    # Set up program for Block 1 and Block 3
    if config.lightColor=='green' and config.lightLevel=='low':
        cfgList = ['1','6', '1', '2', '7','2']
    if config.lightColor=='green' and config.lightLevel=='high':
        cfgList = ['1','b', '1', '2', 'c','2']
    if config.lightColor=='red' and config.lightLevel=='low':
        cfgList = ['1','g', '1', '2','h','2']
    if config.lightColor=='red' and config.lightLevel=='high':
        cfgList = ['1','l', '1', '2','m','2']

    delay = config.lightDelay

    timeList = [delay, config.lightDur, (config.switchDur-(config.lightDur+delay)), delay, config.lightDur, (config.switchDur-(delay + config.lightDur))]

    preCfgList = cfgList
    preTimeList = timeList
    for i in range(config.nFullSwitchPre-1):
        preCfgList = preCfgList + cfgList
        preTimeList = preTimeList + timeList
    preProgramLists = [preCfgList, preTimeList]

    postProgramLists = preProgramLists
    # postCfgList = cfgList
    # postTimeList = timeList
    # for i in range(config.nFullSwitchPost - 1):
    #   postCfgList = postCfgList + cfgList
    #   postTimeList = postTimeList + timeList
    # postProgramLists = [postTimeList, postTimeList]

    return preProgramLists, postProgramLists

def fileOps():
    dt = datetime.now()
    datetimeString = str(dt.month)+"_"+str(dt.day)+"_"+str(dt.year)+"_"+str(dt.hour)+str(dt.minute)
    prePath = config.saveDir + r"\{}\prereinforcement".format(datetimeString)
    rePath = config.saveDir + r"\{}\reinforcement".format(datetimeString)
    postPath = config.saveDir + r"\{}\postreinforcement".format(datetimeString)
    baseDir = dts

    
    os.mkdir(prePath)
    os.mkdir(rePath)
    os.mkdir(postPath)

    return baseDir, prePath, rePath, postPath

def getBG():
    # if address is not None:
    #   bg = cv2.imread(config.bg)
    #   bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
    # else:
    cap = fc2.Context()
    cap.connect(*cap.get_camera_from_index(0))
    cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_15)
    m, f = cap.get_video_mode_and_frame_rate()
    p = cap.get_property(fc2.FRAME_RATE)
    cap.set_property(**p)
    cap.start_capture()


    temp = np.ndarray((480, 640, 60), dtype=np.uint8)

    for i in range(60):

        img = fc2.Image()
        cap.retrieve_buffer(img)
        frame = np.array(img)

        temp[:,:,i] = frame[:,:]

    bgAvg = np.mean(temp, axis=2)

    #_, bg = cv2.threshold(bgAvg, 150, 255, 0)

    bg = bgAvg.astype(np.uint8)


    cap.stop_capture()
    cap.disconnect()

    bgBig = cv2.resize(bg, None, fx = 2, fy = 2)

    r = cv2.selectROI(bgBig, fromCenter=False)

    x, y, w, h = r[0],r[1],r[2],r[3]


    bgBig[y:y+h , x:x+w] = np.ones((h, w), dtype=np.uint8)*160

    bg = cv2.resize(bgBig, None, fx = 0.5, fy = 0.5)

    cv2.destroyAllWindows()


    return bg

if __name__=="__main__":
    baseDir, prePath, rePath, postPath = fileOps()
    preProgramLists, postProgramLists = getProgramLists()

    bg = getBG()

    r = launch_FLIR_GUI(bg)
    test = bg[r[1]:(r[1]+r[3]), r[0]:(r[0]+r[2])]

    _, thresh = cv2.threshold(test, 110, 255, 0)
    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)!=0:
        for contour in contours:
            if cv2.contourArea(contour) > 10000:
                break

    mask = np.zeros(test.shape, np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv2.findNonZero(mask)    

    mask = mask/255
    mask = mask.astype(np.uint8)

    # Pre Reinforcement
    print('Pre-Reinforcement Block')
    saveDir = prePath
    nFrames = int(15*np.sum(preProgramLists[1]))
    wrap = True
    ser = serial.Serial(config.COMM, config.baud)
    time.sleep(2)
    cameraThread = threading.Thread(target = GenericCamera, args=(nFrames, saveDir,'pre',))
    serialThread = threading.Thread(target=GenericSerial, args=(ser, preProgramLists, wrap,'pre',))
    cameraThread.start()
    serialThread.start()
    while (serialThread.isAlive() or cameraThread.isAlive()):
        time.sleep(0.1)
    ser.close()

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
    for i in range(config.nFullSwitchReinforce-1):
        reTimeList = reTimeList + timeList
        reCfgList = reCfgList + cfgList
        reAltCfgList = reAltCfgList + altCfgList

    reProgramLists = [reTimeList, reCfgList, reAltCfgList]
    ser = serial.Serial(config.COMM, config.baud)
    time.sleep(2)
    pcam = ProducerCamera(nFrames = nFrames, saveDir = saveDir, r=r, mask=mask, name='producer')
    cser = ConsumerSerial(ser, reProgramLists, wrap, name='consumer')
    pcam.start()
    cser.start()

    while (pcam.isAlive() or cser.isAlive()):
        time.sleep(0.1)
    ser.write(str.encode('0'))
    ser.close()

    # Post Reinforcement
    print('\nPost-Reinforcement Block')
    saveDir = postPath
    nFrames2 = int(15*np.sum(postProgramLists[1]))
    wrap = True
    ser = serial.Serial(config.COMM, config.baud)
    time.sleep(2)
    cameraThread2 = threading.Thread(target = GenericCamera, args=(nFrames2, saveDir,'post',))
    serialThread2 = threading.Thread(target=GenericSerial, args=(ser, postProgramLists, wrap,'post',))
    cameraThread2.start()
    serialThread2.start()
    while (serialThread2.isAlive() or cameraThread2.isAlive()):
        time.sleep(0.1)
    ser.close()

    print('Processing Pickles')
    for folder in ['prereinforcement', 'reinforcement', 'postreinforcement']:
        
        openDir = baseDir + r'\{}'.format(folder)

        with open(openDir + r'\photos.pkl', 'rb') as f:
            photos = pickle.load(f)

            for i, photo in enumerate(photos):
                frame = np.expand_dims(photo, 2)
                cl_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                #bw_frame = cv2.cvtColor(cl_frame, cv2.COLOR_BGR2GRAY)

                cv2.imwrite(openDir + r"\{}.tiff".format(i), cl_frame)

        os.remove(openDir+r'\photos.pkl')
    print('Done')

    print('Generating Log')
    
    logAddressTxt = dts+r'\log_raw.txt'
    logAddressCsv = logAddressTxt.split('.')[0] + '.csv'

    gg = open(logAddressTxt, mode='r')

    lines = gg.readlines()

    out = open(logAddressCsv.split('.')[0][:-4] + '.csv', mode = 'w')
    outwriter = csv.writer(out, delimiter = ',', quotechar='|', quoting = csv.QUOTE_MINIMAL)

    outwriter.writerow(['blockType', 'experimentTime', 'xPos', 'yPos', 'flowDir', 'lightState'])

    for line in lines:
        elements = line.split(' ')
        elements2 = elements[2].split(',')
        elements2.insert(0, elements[0])

        if elements2[0] in ['(consumer', '(producer']:
            block = 'reinforcement'
        else:
            threadNo = int(elements2[0].split('-')[1])

            if threadNo < 3:
                block = 'prereinforcement'
            elif threadNo > 3:
                block = 'postreinforcement'

        time = elements2[1]

        if len(elements2)==6 or len(elements2)==4:
            #Anchor row
            if block in ['prereinforcement', 'postreinforcement']:
                x = None
                y = None
                count = elements2[-1][:-2]

            else:
                x = elements2[-2]
                y = elements2[-1][:-2]
                count = elements2[-1][-2]

            fromBuffer = serialBuffer[-1]

            dir = fromBuffer[0][-1]
            light = fromBuffer[1]

            row = [block, time, x, y, dir, light]
            cleanRow = [str(e).rstrip() for e in row]
            outwriter.writerow(cleanRow)

        elif len(elements2) == 5:
            serialBuffer = []

            dir = elements2[-2].split('::')[1][-1]
            light = (elements2[-1].split('::')[1]).split('\n')[0]

            serialBuffer.append([dir, light])



    # out.close()
    # os.remove(logAddressTxt)
    print('Log Complete and stored at {}'.format(logAddressCsv))

    

