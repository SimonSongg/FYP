#!/usr/bin/env python3
import serial
import os
import rospy
from std_msgs.msg import Float64MultiArray

os.system('echo %s | sudo -S %s' % ('20001007', 'chmod 777 /dev/ttyUSB1'))

ser_front = serial.Serial("/dev/ttyUSB1",9600,timeout=0.3)


state = 0 
data_256_1 = 0
data_16_1 = 0
data_256_2 = 0
data_16_2 = 0
data_256_3 = 0
data_16_3 = 0
data_256_4 = 0
data_16_4 = 0
counter = 0

data_list = [0.0, 0.0, 0.0, 0.0]

pubF = rospy.Publisher('UltraDistanceFront', Float64MultiArray, queue_size = 5)

rospy.init_node('Ultrasonic_combo_Publisher', anonymous = False)

#ultra_dataF = Float64MultiArray()
#ultra_dataR = Float64MultiArray()
ser_front.write(bytes.fromhex('00'))
while not rospy.is_shutdown():


    
    line = ser_front.read(1)
    if (line.hex() == ''):
        ser_front.write(bytes.fromhex('00'))
        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        continue

    dataF = int.from_bytes(line,byteorder='little')
    print(dataF)


    if dataF == 255:
        if state == 0:
            state = 1
        elif state == 1:
            state = 10
        elif state == 2:
            data_16_1 = dataF
            state = 3
        elif state == 10:
            data_16_1 = 255
            data_256_1 = 255
            #data_list[0] = 0.0
            state = 3
        elif state == 3:
            state = 20
        elif state == 4:
            data_16_2 = dataF
            state = 5
        elif state == 20:
            data_16_2 = 255
            data_256_2 = 255
            #data_list[1] = 0.0
            state = 5
        elif state == 5:
            state = 30
        elif state == 6:
            data_16_3 = dataF
            state = 7
        elif state == 30:
            data_16_3 = 255
            data_256_3 = 255
            #data_list[2] == 0.0
            state = 7
        elif state == 7:
            state = 40
        elif state == 8:
            data_16_4 = dataF
            state = 9
        elif state == 40:
            data_16_4 = 255
            data_256_4 = 255
            #data_list[3] = 0.0
            state = 9
        elif state == 9:
            if ((255 + data_256_1 + data_256_2 + data_256_3 + data_256_4 + data_16_1 + data_16_2 + data_16_3 + data_16_4) & 255 == dataF):
                data_list[0] = data_256_1 * 256 + data_16_1
                data_list[1] = data_256_2 * 256 + data_16_2
                data_list[2] = data_256_3 * 256 + data_16_3
                data_list[3] = data_256_4 * 256 + data_16_4

                pubdata = Float64MultiArray(data = data_list)
                pubF.publish(pubdata)
                data_list = [0.0, 0.0, 0.0, 0.0]
                state = 0 
                data_256_1 = 0
                data_16_1 = 0
                data_256_2 = 0
                data_16_2 = 0
                data_256_3 = 0
                data_16_3 = 0
                data_256_4 = 0
                data_16_4 = 0
            else:
                print("error msg received")
                state = 0 
                data_list = [0.0, 0.0, 0.0, 0.0]
                data_256_1 = 0
                data_16_1 = 0
                data_256_2 = 0
                data_16_2 = 0
                data_256_3 = 0
                data_16_3 = 0
                data_256_4 = 0
                data_16_4 = 0
    else:
        if state == 1:
            data_256_1 = dataF
            state = 2
        elif state == 2:
            data_16_1 = dataF
            state = 3
        elif state == 3:
            data_256_2 = dataF
            state = 4
        elif state == 4:
            data_16_2 = dataF
            state = 5
        elif state == 5:
            data_256_3 = dataF
            state = 6
        elif state == 6:
            data_16_3 = dataF
            state = 7
        elif state == 7:
            data_256_4 = dataF
            state = 8
        elif state == 8:
            data_16_4 = dataF
            state = 9
        elif state == 9:
            if ((255 + data_256_1 + data_256_2 + data_256_3 + data_256_4 + data_16_1 + data_16_2 + data_16_3 + data_16_4) & 255 == dataF):
                data_list[0] = data_256_1 * 256 + data_16_1
                data_list[1] = data_256_2 * 256 + data_16_2
                data_list[2] = data_256_3 * 256 + data_16_3
                data_list[3] = data_256_4 * 256 + data_16_4

                pubdata = Float64MultiArray(data = data_list)
                pubF.publish(pubdata)
                data_list = [0.0, 0.0, 0.0, 0.0]
                state = 0 
                data_256_1 = 0
                data_16_1 = 0
                data_256_2 = 0
                data_16_2 = 0
                data_256_3 = 0
                data_16_3 = 0
                data_256_4 = 0
                data_16_4 = 0
            else:
                print("error msg received")
                state = 0 
                data_list = [0.0, 0.0, 0.0, 0.0]
                data_256_1 = 0
                data_16_1 = 0
                data_256_2 = 0
                data_16_2 = 0
                data_256_3 = 0
                data_16_3 = 0
                data_256_4 = 0
                data_16_4 = 0
        elif state == 10 or state == 20 or state == 30 or state ==40:
            print("error msg received")
            state = 0
            data_list = [0.0, 0.0, 0.0, 0.0]
            data_256_1 = 0
            data_16_1 = 0
            data_256_2 = 0
            data_16_2 = 0
            data_256_3 = 0
            data_16_3 = 0
            data_256_4 = 0
            data_16_4 = 0
    counter = counter + 1
    if (counter == 10):
        ser_front.write(bytes.fromhex('00'))
        ser_front.write(bytes.fromhex('00'))
        ser_front.write(bytes.fromhex('00'))
        ser_front.write(bytes.fromhex('00'))
        #ser_front.write(bytes.fromhex('00'))

        counter =0