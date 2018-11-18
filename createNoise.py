import csv
from shutil import copyfile


categories = [
    'door_wood_knock'
]

noise_files_list = []

with open('/home/ankur/Downloads/Others/ESC-50-master/meta/esc50.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        category = row[3]

        if category in categories:
            noise_files_list.append((row[0], category))

raw_folder = '/home/ankur/Downloads/Others/ESC-50-master/audio'
dest_folder = '/home/ankur/Downloads/Others/noise'

for noise_file in noise_files_list:
    noise_file_path = raw_folder + '/' + noise_file[0]
    dest_file_path = dest_folder + '/' + noise_file[1] + '_' + noise_file[0]
    copyfile(noise_file_path, dest_file_path)

print("Done")