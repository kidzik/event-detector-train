import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import btk
import re
import os

input_dir = "/home/lukasz/Dropbox/tempdeid/"
output_dir = "csv"

def extract_kinematics(leg, filename):
    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader() 
    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()

    metadata = acq.GetMetaData()

    rate = int(metadata.FindChild('POINT').value().FindChild('RATE').value().GetInfo().ToDouble()[0])

    # We extract only kinematics
    kinematics = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles", "FootProgressAngles"]

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
    angles = [None] * len(kinematics)
    for i, v in enumerate(kinematics):
        point = acq.GetPoint(leg + v)
        angles[i] = point.GetValues()
        
    curves = np.concatenate(angles, axis=1)

    # Plot each component of the big array
    # for i in range(3 * len(kinematics)):
    #     plt.plot(range(nframes), curves[:,i])

    # Add events as output
    for event in btk.Iterate(acq.GetEvents()):
        if event.GetFrame() >= nframes:
            return
        if event.GetContext()[0] == leg:
            if event.GetLabel() == "Foot Strike":
                outputs[event.GetFrame(), 0] = 1
            elif event.GetLabel() == "Foot Off":
                outputs[event.GetFrame(), 1] = 1
            
    if (np.sum(outputs) == 0):
        print("No events in %s!" % (filename,))
        return

    arr = np.concatenate((curves, outputs), axis=1)

    m = re.match(input_dir + "(?P<name>.+).c3d",filename)
    name = m.group('name').replace(" ","-")
    np.savetxt("%s/%s%s.csv" % (output_dir, leg, name), arr, delimiter=',')

# Extract kinematics from all *.c3d files in c3d directory
files = os.listdir(input_dir)
for filename in files:
    for leg in ['L','R']:
        extract_kinematics(leg, input_dir + filename)
