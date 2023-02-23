# File to convert XML into a useable TXT format

from xml.etree import ElementTree as ET
import os 

# TODO: Make storing files fully modular 

# Storing video file location
video_file = "B:/CapstoneData/Datasets/video_0001.xml"
x = 0

tree = ET.parse(video_file) # Parse the XML file
annotations = tree.getroot() # Get the root of the XML file

for child in annotations:
    # print (child.tag, child.attrib)
    if (child.get('label') == 'pedestrian' or child.get('label') == 'ped'):
        for attribute in child:
            # If frame number is less than 99 add a 000 to the front of the frame number. If frame number is less than 9 add 0000, else add a 00
            if (int(attribute.get('frame')) < 10):
                buffer = '0000'
            elif (int(attribute.get('frame')) < 100):
                buffer = '000'
            else:
                buffer = '00'

            xCenter = (float(attribute.get('xbr')) + float(attribute.get('xtl'))) / 2
            yCenter = (float(attribute.get('ybr')) + float(attribute.get('ytl'))) / 2
            width = float(attribute.get('xbr')) - float(attribute.get('xtl'))
            height = float(attribute.get('ybr')) - float(attribute.get('ytl'))
            xCenter = xCenter / 1920
            width = width / 1920
            yCenter = yCenter / 1080
            height = height / 1080

            # Make sure you only run the file once
            if (os.path.exists("B:/CapstoneData/ParsedFiles/" + buffer + attribute.get('frame') + ('.txt'))):
                with open("B:/CapstoneData/ParsedFiles/" + buffer + attribute.get('frame') + ('.txt'), 'a') as f:
                    f.write('0' + " " + str(xCenter) + " " + str(yCenter) + " " + str(width) + " " + str(height) + '\n')
                    # f.write('0' + " " + attribute.get('xtl') + " " + attribute.get('ytl') + " " + attribute.get('xbr') + " " + attribute.get('ybr') + '\n')
            else:
                with open("B:/CapstoneData/ParsedFiles/" + buffer + attribute.get('frame') + ('.txt'), 'w') as f:
                    f.write('0' + " " + str(xCenter) + " " + str(yCenter) + " " + str(width) + " " + str(height) + '\n')

                    # f.write('0' + " " + attribute.get('xtl') + " " + attribute.get('ytl') + " " + attribute.get('xbr') + " " + attribute.get('ybr') + '\n')
