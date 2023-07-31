################### SPLIT DATA FOR FINE TUNE ########################
import os
import random
import shutil

# Set the path to the main folder
main_folder = r'C:\pythonProject\pythonProjectHF\dataset'

# Set the path to the train and dev folders
train_folder = './train'
dev_folder = './dev'

# Set the split ratio
split_ratio = 0.8  # 80% for train, 20% for dev

# Get a list of all files in the main folder
files = os.listdir(main_folder)

# Separate WAV and TXT files
wav_files = [file for file in files if file.endswith('.wav')]
txt_files = [file for file in files if file.endswith('.txt')]

# Shuffle the list of WAV files
random.shuffle(wav_files)

# Calculate the split index
split_index = int(split_ratio * len(wav_files))

# Split the WAV files into train and dev lists
train_wav_files = wav_files[:split_index]
dev_wav_files = wav_files[split_index:]

# Create the train and dev folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(dev_folder, exist_ok=True)

# copy the WAV files to the train folder
for wav_file in train_wav_files:
    src_wav = os.path.join(main_folder, wav_file)
    dst_wav = os.path.join(train_folder, wav_file)
    shutil.copy(src_wav, dst_wav)

# copy the WAV files to the dev folder
for wav_file in dev_wav_files:
    src_wav = os.path.join(main_folder, wav_file)
    dst_wav = os.path.join(dev_folder, wav_file)
    shutil.copy(src_wav, dst_wav)

# copy the corresponding TXT files to the respective folders
for txt_file in txt_files:
    src_txt = os.path.join(main_folder, txt_file)
    if txt_file.replace('.txt', '.wav') in train_wav_files:
        dst_txt = os.path.join(train_folder, txt_file)
        shutil.copy(src_txt, dst_txt)
    elif txt_file.replace('.txt', '.wav') in dev_wav_files:
        dst_txt = os.path.join(dev_folder, txt_file)
        shutil.copy(src_txt, dst_txt)

print("Folder split completed!")


######################## TAR.GZ FILE ############################
import tarfile
import os

def compress_folder_to_tar_gz(folder_path, output_path):
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))

folder_to_compress = r"C:\pythonProject\pythonProjectHF\dataset\dev"
output_tar_gz_file = r"C:\pythonProject\pythonProjectHF\dataset\dev.tar.gz"

compress_folder_to_tar_gz(folder_to_compress, output_tar_gz_file)

########################### CSV FILE ###############################33
import os
import csv

# Set the path to the folder containing the WAV and TXT files
folder_path = r'C:\Users\User\Desktop\TryDataset'

# Set the path for the CSV file
csv_file = r'C:\Users\User\Desktop\TryDataset\dataset.csv'

# Initialize an empty list to store the rows of the CSV
csv_rows = []

# Iterate through the WAV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav'):
        wav_file = file_name
        txt_file = file_name[:-4] + '.txt'

        # Read the content of the TXT file with the correct encoding
        with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as txt:
            txt_content = txt.read().strip()

        csv_rows.append([wav_file, txt_content])

# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['audio', 'trancription'])  # Write the header row
    writer.writerows(csv_rows)