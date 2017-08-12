from utils import *
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import btk
import re
import os
import keras
from keras.models import load_model

def derivative(traj, nframes):
    traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
    return np.append(traj_der, [[0,0,0]], axis=0)

def extract_kinematics(leg, filename_in, filename_out):
    print("Trying %s" % (filename_in))
    
    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader() 
    reader.SetFilename(filename_in)
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()
    first_frame = acq.GetFirstFrame()

    metadata = acq.GetMetaData()

    rate = int(metadata.FindChild('POINT').value().FindChild('RATE').value().GetInfo().ToDouble()[0])
    if rate != 120:
        return

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles", "FootProgressAngles"]
    markers = ["ANK", "TOE", "KNE", "ASI", "HEE"]
    
    # Cols
    # 2 * 5 * 3 = 30  kinematics
    # 2 * 5 * 3 = 30  marker trajectories
    # 2 * 5 * 3 = 30  marker trajectory derivatives
    # 3 * 3 = 9       extra trajectories

    outputs = np.array([[0] * nframes, [0] * nframes]).T
    
    # Check if there are any kinematics in the file
    brk = True
    for point in btk.Iterate(acq.GetPoints()):
        if point.GetLabel() == "L" + kinematics[0]:
            brk = False
            break

    if brk:
        print("No kinematics in %s!" % (filename,))
        return

    # Combine kinematics into one big array
    opposite = {'L': 'R', 'R': 'L'}
    angles = [None] * (len(kinematics) * 2)
    for i, v in enumerate(kinematics):
        point = acq.GetPoint(leg + v)
        angles[i] = point.GetValues()
        point = acq.GetPoint(opposite[leg] + v)
        angles[len(kinematics) + i] = point.GetValues()
    
    # Get the pelvis
    LASI = acq.GetPoint("LASI").GetValues()
    RASI = acq.GetPoint("RASI").GetValues()
    midASI = (LASI + RASI) / 2
    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    traj = [None] * (len(markers) * 4 + 3)
    for i, v in enumerate(markers):
        try:
            traj[i] = acq.GetPoint(leg + v).GetValues() - midASI
            traj[len(markers) + i] = acq.GetPoint(opposite[leg] + v).GetValues() - midASI
        except:
            return

        traj[i][:,0] = traj[i][:,0] #* incrementX
        traj[len(markers) + i][:,0] = traj[len(markers) + i][:,0] #* incrementX
        traj[i][:,2] = traj[i][:,2] #* incrementX
        traj[len(markers) + i][:,2] = traj[len(markers) + i][:,2] #* incrementX

    for i in range(len(markers)*2):
        traj[len(markers)*2 + i] = derivative(traj[i], nframes) 
        
    midASI = midASI #* incrementX

    midASIvel = derivative(midASI, nframes)
    midASIacc = derivative(midASIvel, nframes)

    traj[len(markers)*4] = midASI
    traj[len(markers)*4 + 1] = midASIvel
    traj[len(markers)*4 + 2] = midASIacc

    curves = np.concatenate(angles + traj, axis=1)

    # Plot each component of the big array
    # for i in range(3 * len(kinematics)):
    #     plt.plot(range(nframes), curves[:,i])

    # Add events as output
    for event in btk.Iterate(acq.GetEvents()):
        if event.GetFrame() >= nframes:
            print("Event happened to far")
            return
        if len(event.GetContext()) == 0:
            print("No events")
            return
        if event.GetContext()[0] == leg:
            if event.GetLabel() == "Foot Strike":
                outputs[event.GetFrame() -first_frame, 0] = 1
            elif event.GetLabel() == "Foot Off":
                outputs[event.GetFrame() - first_frame, 1] = 1
            
    if (np.sum(outputs) == 0):
        print("No events in %s!" % (filename,))
        return

    arr = np.concatenate((curves, outputs), axis=1)

    print("Writig %s" % filename_out)
    np.savetxt(filename_out, arr, delimiter=',')

parser = argparse.ArgumentParser(description='Annotate heel strike (HS) and foot off (FO) events.')
parser.add_argument('event', metavar='event', help="Type of the event to predict: heel strike (HS) or foot-off (FO)", type=str, choices=['HS','FO'])
parser.add_argument('filename', metavar='filename', help="A c3d file with kinematics", type=str)

# parser.add_argument('--method', '-m',
#                     help="",
#                     type=str, choices=['neural','velocity','coordinate'],default="neural")
args = parser.parse_args()

def convert_data(data):
    # TODO: temporary mess
    def derivative(traj):
        nframes = traj.shape[0]
        traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
        return np.append(traj_der, [[0] * traj.shape[1]], axis=0)

    # We assume the data has following sequences
    # joint angles (3 DOF each):
    # - hip
    # - knee
    # - ankle
    # - pelvis
    # - foot progression
    # markers positions (3 DOF each):
    # - ankle
    # - toes
    # - knee
    # - pelvis
    # - heel
    # if there are two extra columns we assume that these are binary
    # sequences of heel strike and foot off events respectively
    if data.shape[1] != 30 and data.shape[1] != 32:
        sys.exit("Wrong data format. There should be 30 columns.")

    # What we actually use are:
    # - joint angles (5 x 3)
    # - velocity of markers (5 x 3)
    # - velocity and acceleration of the pelvis (2 x 3)
    X = np.zeros( (data.shape[0], 15 + 15 + 6) )
    X[:,0:15] = data[:,0:15]
    X[:,15:30] = derivative(data[:,15:30])
    X[:,30:33] = derivative(data[:,24:27])
    X[:,33:36] = derivative(derivative(data[:,24:27]))

    Y = None
    if data.shape[1] != 32:
        Y = data[:,30:32]

    return X, Y

extract_kinematics('L', args.filename, "tmp.csv")

def neural_method(inputs):
    keras.losses.weighted_binary_crossentropy = weighted_binary_crossentropy
    model = load_model("models/%s.h5" % args.event)
    cols = range(15) + [15 + i for i in range(13)] + [30 + i for i in range(6)] 
    res = model.predict(inputs[:,cols].reshape((1,inputs.shape[0],len(cols))))
    peakind = peakdet(res[0], 0.7)
    print ', '.join(map(str, [k for k,v in peakind[0]]))

inputs = np.loadtxt("tmp.csv", delimiter=',')
idxL = [(i / 3) * 3 + i  for i in range(30)]
idxR = [3 + (i / 3) * 3 + i  for i in range(30)]
inputsL = inputs[:, idxL]
inputsR = inputs[:, idxR]
XL, YL = convert_data(inputsL)
XR, YR = convert_data(inputsR)

neural_method(XL)
