#!/usr/bin/env python3

import rospy
import os

from ctypes import *

import _ctypes

import ctypes





CUR_PATH=os.path.dirname(__file__)

dllPath=os.path.join(CUR_PATH,"libtest.so")

pDll=ctypes.cdll.LoadLibrary(dllPath)

result=0



class UWBMsg(Structure):

     _fields_ = [("x", c_double),

                ("y", c_double),

                ("z", c_double)]





location = UWBMsg()

anchorArray=(UWBMsg*3)()

distanceArray=(c_int*3)()







anchorArray[0].x=0

anchorArray[0].y=0

anchorArray[0].z=2



anchorArray[1].x=0

anchorArray[1].y=10

anchorArray[1].z=2



anchorArray[2].x=10

anchorArray[2].y=10

anchorArray[2].z=2





distanceArray[0]=7100

distanceArray[1]=7100

distanceArray[2]=7100



result=pDll.GetLocation(byref(location),0,anchorArray,distanceArray)

                    

print(location.x)

print(location.y)

print(location.z)

               

print(result)