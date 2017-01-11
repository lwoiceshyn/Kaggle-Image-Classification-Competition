import os
import os.path
import shutil
import csv 

folder_path = "~/Downloads/A3/train"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

with open('/home/asblab/Downloads/A3/train.csv', 'rb') as f
	dialect = csv.Sniffer().sniff(csvfile.read(), delimiters=',')
	csvfile.seek(0)
	reader = csv.reader(f, dialect)

	for line in reader:
		print line

