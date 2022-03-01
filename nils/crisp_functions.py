"""
Some functions for Sergej to use
"""
import numpy as np
import pydoocs
import time
import sys

"""
-------------------------------HELPER FUNCTIONS----------------------------------------------------------- 
"""


stage_motors_adresse = 'XFEL.SDIAG/SPS/'
stage_motors = np.array(['MOTOR5.1935.TL', 'MOTOR6.1935.TL', 'MOTOR7.1935.TL', 'MOTOR8.1935.TL', 'MOTOR9.1935.TL'])


def get_stage_position():
    outcome = np.zeros(np.size(stage_motors))
    for n in np.arange(np.size(stage_motors)):
        outcome [n] = pydoocs.read(stage_motors_adresse + stage_motors[n] + '/FPOS') ['data']
    if np.sum(outcome)>5*110 - 3:
        out = 'high'
    elif np.sum(outcome)<3:
        out = 'low'
    else:
        out = 'undefined'
    return out

def push_stage_position(input_string):

    if input_string == 'high':
        to_set = 110
    elif input_string == 'low':
        to_set = 0
    else:
        print('Instruction to move filter stages unclear, I quit!')
        return None

    for n in np.arange(np.size(stage_motors)):
        #wert setzen
        pydoocs.write(stage_motors_adresse + stage_motors[n] + '/FPOS.SET', to_set)
        time.sleep(0.5)
        # sicher gehen dass motor nicht schon wiei fährt
        while int(format(pydoocs.read(stage_motors_adresse + stage_motors[n] + '/MSTATUS')['data'],'06b') [-3]) !=1:
            time.sleep(0.5)
            sys.stdout.write("\rwaiting for motors to finish previous move" )
            sys.stdout.flush()
        #wert fahren
        pydoocs.write(stage_motors_adresse + stage_motors[n] + '/CMD', 1)
        time.sleep(0.5)
    print('Motors set to move')

    #only quit when motorshave reached the goal
    while get_stage_position () != input_string:
            time.sleep(1)
            sys.stdout.write("\rMotors moving" )
            sys.stdout.flush()
    sys.stdout.write("Motors finished" )
    sys.stdout.flush()
    return None
"""
-----------------------REQUESTED FUNCTIONS ---------------------------------------------
"""
def update_crisp():
    """
    moves the gratings back and fourth so the form factor server updates along the entire frequency range
    """
    #check which on is currently in:
    currently = get_stage_position()
    if currently == 'low':
        push_stage_position('high')
        time.sleep(5)
    else:
        push_stage_position('low')
        time.sleep(5)
        push_stage_position('high')

    return None

def select_bunch(bunch_number):
    """
    writes bunch number to form factor server, so that the correct one is selected
    """
    pydoocs.write('XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/NTH_BUNCH', bunch_number)
    time.sleep(1)
    selected_bunch = pydoocs.read('XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/NTH_BUNCH')['data']
    return selected_bunch

def get_reconstruction():
    """
    get time(fs) vs current(A) from reconstruction server. Be careful it reconstructed from an average of 16 maropulses
    """
    data = pydoocs.read('XFEL.SDIAG/THZ_SPECTROMETER.RECONSTRUCTION/CRD.1934.TL/OUTPUT_TIMES')
    current = pydoocs.read('XFEL.SDIAG/THZ_SPECTROMETER.RECONSTRUCTION/CRD.1934.TL/CURRENT_PROFILE')['data']
    time = data ['data']
    return [time, current]
