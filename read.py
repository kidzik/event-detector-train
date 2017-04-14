import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import btk
import re
import os

input_dir = "/media/lukasz/TOSHIBA EXT/tempdeidNEW/" 
output_dir = "/media/lukasz/TOSHIBA EXT/csv-hee/" 

def derivative(traj, nframes):
    traj_der = traj[1:nframes,:] - traj[0:(nframes-1),:]
    return np.append(traj_der, [[0,0,0]], axis=0)

def extract_kinematics(leg, filename):
    m = re.match(input_dir + "(?P<name>.+).c3d",filename)
    name = m.group('name').replace(" ","-")
    output_file = "%s/%s%s.csv" % (output_dir, leg, name)
    print("Trying %s" % (filename))
    
    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader() 
    reader.SetFilename(filename)
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()
    first_frame = acq.GetFirstFrame()

    if os.path.isfile(output_file):
        return

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

    print("Writig %s" % filename)
    np.savetxt(output_file, arr, delimiter=',')

# Extract kinematics from all *.c3d files in c3d directory
files = os.listdir(input_dir)
for filename in files:
    for leg in ['L','R']:
        extract_kinematics(leg, input_dir + filename)
        print(filename)
