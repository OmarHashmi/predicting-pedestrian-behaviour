# File to convert XML into a useable TXT format

from xml.etree import ElementTree as ET
import os 

# Loop through all the files in the folder
annotations_dir = "annotations/"

for filename in os.listdir(annotations_dir):
	if filename.endswith(".xml"):
		video_file = os.path.join(annotations_dir, filename)
	else:
		continue

	x = 0

	tree = ET.parse(video_file) # Parse the XML file
	annotations = tree.getroot() # Get the root of the XML file

	unique_id = annotations[1][0][1].text
	print(unique_id)

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

				# Create a new folder within ParsedFiles to store the new files with a unique_id
				# new_folder_name = video_file.split('/')[-1].split('.')[0]
				new_folder_path = os.path.join("ParsedFiles")
				if not os.path.exists(new_folder_path):
					os.mkdir(new_folder_path)

				# Make sure you only run the file once and place the files in the correct folder
				if (os.path.exists(os.path.join(new_folder_path, unique_id + buffer + attribute.get('frame') + ('.txt')))):
					with open(os.path.join(new_folder_path, unique_id + buffer + attribute.get('frame') + ('.txt')), 'a') as f:
						f.write('0' + " " + str(xCenter) + " " + str(yCenter) + " " + str(width) + " " + str(height) + '\n')
						# f.write('0' + " " + attribute.get('xtl') + " " + attribute.get('ytl') + " " + attribute.get('xbr') + " " + attribute.get('ybr') + '\n')
				else:
					with open(os.path.join(new_folder_path, unique_id + buffer + attribute.get('frame') + ('.txt')), 'w') as f:
						f.write('0' + " " + str(xCenter) + " " + str(yCenter) + " " + str(width) + " " + str(height) + '\n')
						# f.write('0' + " " + attribute.get('xtl') + " " + attribute.get('ytl') + " " + attribute.get('xbr') + " " + attribute.get('ybr') + '\n')