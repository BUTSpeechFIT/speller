#!/usr/bin/python

import sys
import os
import torch
import numpy as np
#----------------------------------------
def Teacherforce_schedule(args):
        "This takes almost linear steps to get to the minimum value"
        t=0 ; Tc=1; To=args.teacher_force
        rate=3/args.teacher_force_decay_rate
        while True:
                t+=1 
                yield To+(Tc-To)*np.exp(-rate*t)
#------------------------------




#teacher_force_decay_rate=10
#teacher_force=0.6
# #----------------------------------------
#def Teacherforce_schedule_linear(teacher_force_decay_rate,teacher_force):
#         "This takes almost linear steps to get to the minimum value"
#         t=0 ; Tc=1; To=teacher_force
#         rate=1/teacher_force_decay_rate
#         while True:
#                 t+=1 
#                 yield To+(Tc-To)*np.exp(-rate*t)
#                yield To + max(Tc - Tc*rate*t,0)
# #------------------------------


#maxiterations=1001
#gen=Teacherforce_schedule_linear(teacher_force_decay_rate,teacher_force)

#for i in range(1,maxiterations):
#        TF=next(gen)
#        if TF<0:
#                exit(0)
#        print(i,TF) 


