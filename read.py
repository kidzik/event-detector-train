import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import btk

filename = "c3d/223000 10.c3d"

# Open c3d and read data
reader = btk.btkAcquisitionFileReader() 
reader.SetFilename(filename)
reader.Update()
acq = reader.GetOutput()

# We extract only kinematics
kinematics = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles", "FootProgressAngles"]

# Combine kinematics into one big array
angles = [None] * len(kinematics)
for i, v in enumerate(kinematics):
    point = acq.GetPoint("L" + v)
    angles[i] = point.GetValues()

curves = np.concatenate(angles, axis=1)

# Plot each component of the big array
for i in range(3 * len(kinematics)):
    plt.plot(range(acq.GetPointFrameNumber()), curves[:,i])

# Plot the events
for event in btk.Iterate(acq.GetEvents()):
    if event.GetContext() == "Left":
        col = "r"
        if event.GetLabel() == "Foot Strike":
            col = "b"
        plt.axvline(x=event.GetFrame(), color = col)

plt.xlabel('Frame #')
plt.ylabel('Joint angle')
plt.title('Joint angles in time and events')
plt.show()
