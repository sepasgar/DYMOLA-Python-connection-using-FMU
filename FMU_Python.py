# Author: Sepehr Asgari (MSc. Simulation Science @ RWTH AACHEN)
# Contact: sepehrasgari.re@gmail.com

'''
This code is a clear and compact script to connect DYMOLA or in general a Functional Mock-up Unit (FMU) to a Python script using the 'fmpy' library.
This is particularly useful when you want to make use of a Machine/Deep learning model in your simulation to improve its performance.
To install the 'fmpy', simply call 'pip install FMPy' in your terminal.
'''

# IMPORT YOUR LIBRARIES
##################
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import datetime
import time  
import os
import joblib
##################

# LIBRARY NEED TO DEAL WITH FMU ("FMPY")
# Note: There are also other libraries to deal with FMU but with chose FMPY due to easiness of installation. You can download it from PyPi.
##################
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from fmpy.util import plot_result, download_test_file
from fmpy import dump
##################


##################
# DEFINE A FUNCTION (OPTIONAL BUT RECOMMENDED)
def test_sim():

    print('Starting LR co-simulation')

    # LOAD YOUR PRE-TRAINED ML MODEL 
    model_path = 'PATH_TO_MODEL/YOUR_MODEL.pkl' 
    model = joblib.load(model_path)

    # LOAD YOUR FMU (YOU SHOULD SAVE YOUR MODEL IN FMU FORMAT IN DYMOLA)
    fmu_filename = 'PATH_TO_YOUR_FMU/YOUR_FMU.fmu'

    # DEFINE THE SIMULATION TIME, YOU CAN ALSO DEFINE OTHER SIMULATION PARAMETERS HERE
    start_time = 0       #SECOND
    stop_time = 100000   #SECOND
    step_size = 60       #SECOND

    # READ MODEL DESCRIPTION
    model_description = read_model_description(fmu_filename)

    # COLLECT THE VARIABLES (VARIABLES ARE DEFINED IN DYMOLA)
    vrs = {variable.name: variable.valueReference for variable in model_description.modelVariables}

    # GETTING THE VALUE REFERENCES FOR THE VARIABLES YOU WANT TO GET/SET
    vr_inputs = vrs['YOUR_INPUT'] 
    vr_outputs = [vrs['YOUR_OUTPUT1'], vrs['YOUR_OUTPUT2'], vrs['YOUR_OUTPUT3']]

    # EXTRACT FMU
    unzipdir = extract(fmu_filename)

    # PUT FMU INTO WORK
    fmu = FMU2Slave(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.coSimulation.modelIdentifier,
                    instanceName='instance1')

    # INITIALIZE FMU
    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    # INITIALIZE OUTPUTS LISTS
    # Initialize output lists
    OUTPUT1, OUTPUT2, OUTPUT3, OUTPUT4, OUTPUT5, OUTPUT6, OUTPUT7 = [], [], [], [], [], [], []

    print('Co-simulation started', datetime.datetime.now())

    # STARTING SIMULATION LOOP -> FMU IS SENDING THE DATA FROM DYMOLA TO PYTHON--
    #                 --AND AFTER YOUT ML/DL MODEL PREDICTS THE VARIABLE, THEY WILL BE SENT BACK TO DYMOLA TO DO THE SIMULATION
    loop_start_time = time.time()

    for time_ in range(start_time, stop_time, step_size):

        #print(time_)

        # RETRIEVE DATA FROM DYMOLA IF YOU NEED THEM AS INPUT TO THE ML MODEL (OPTIONAL)
        t_out = fmu.getReal([vr_outputs[5]])[0]
        t_in = fmu.getReal([vr_outputs[1]])[0]

        # PREPARE DATA TO INPUT THE ML MODEL
        X = np.array([[t_in-273.0, t_out-273.0]]) # FOR EXAMPLE IF THEY ARE TEMPERATURE IN KELVIN

        # GET PREDICTION FROM THE ML MODEL
        preds_eval = model.predict(X)
        print(X, preds_eval)

        # SET THE STATE OF THE VARIABLE BASED ON THE ML PREDICTION (IF > 50% -> SET TO INPUT TO 0, ELSE SET IT TO 1)
        if preds_eval[0] > 0.5:
            fmu.setReal([vr_inputs], [0]) # IF THERE ARE MULTIPLE INPUTS USE [vr_inputs[i]]
        else:
            fmu.setReal([vr_inputs], [1])

        # PERFORM A SIMULATION STEP
        fmu.doStep(currentCommunicationPoint=time_, communicationStepSize=step_size)

        # RETRIEVE OTHER OUTPUTS IF NEEDED
        co2_out = fmu.getReal([vr_outputs[2]])[0]
        tzul_out = fmu.getReal([vr_outputs[3]])[0]
        ventilation_out = fmu.getReal([vr_outputs[4]])[0]
        t_simulated = fmu.getReal([vr_outputs[0]])[0]

        # APPEND THE VALUES TO THEIR LISTS
        OUTPUT1.append(preds_eval[0])
        OUTPUT2.append(t_simulated)
        OUTPUT3.append(t_in)
        OUTPUT4.append(co2_out)
        OUTPUT5.append(tzul_out)
        OUTPUT6(ventilation_out)
        OUTPUT7.append(t_out)  

    loop_end_time = time.time()

    loop_duration = loop_end_time - loop_start_time
    print(f'Loop duration: {loop_duration:.2f} seconds')

    print('Co-simulation finished', datetime.datetime.now())

    #  STACK ALL THE OUTPUTS
    all_data = np.vstack([
        OUTPUT1,
        OUTPUT2,
        OUTPUT3,
        OUTPUT4,
        OUTPUT5,
        OUTPUT6,
        OUTPUT7  
    ])

    # SAVE THE COMBINED DATASET IN .mat FORMAT
    sio.savemat('PATH_TO_YOUR_DIRECTORY/results.mat', {'RESULTS': all_data})
##################

# CALL THE FUNCTION
test_sim()
print('Test finished', datetime.datetime.now())
