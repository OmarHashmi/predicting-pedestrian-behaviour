# File to convert 
import os 
import shutil

# Loop through all the files in the folder
images_dir = "B:\CapstoneData\Datasets\JAADset\JAAD\images"
common_dir = "B:\CapstoneData\imagesDir"

# Loop through all the folders in the folder
for videoName in os.listdir(images_dir):
    unique_id = videoName

    # Loop through all the files in the folder
    for filename in os.listdir(os.path.join(images_dir, videoName)):
        os.rename(os.path.join(images_dir, videoName, filename), os.path.join(images_dir, videoName, unique_id + filename))

    for renamed_file in os.listdir(os.path.join(images_dir, videoName)):
        shutil.move(os.path.join(images_dir, videoName, renamed_file), os.path.join(common_dir, renamed_file))