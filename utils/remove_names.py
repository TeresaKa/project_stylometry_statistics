import glob
import os


path = "*.txt"
#prefix = 'French/'

for file in glob.glob(path):
    print(file)
    filename = file.replace(',', '_')
    #f = file.replace(prefix, '')
    #print(f)
    os.rename(file, filename)
