import os,shutil,csv
from tkinter.constants import FALSE
import pandas as pd

def find_files(path,filename):
   data_csv= pd.read_csv(filename)
   for root, dirs, files in os.walk(path):
      data_img = pd.DataFrame(files,columns=['ImageId'])
   dfs_dictionary = {'DF1': data_csv.ImageId , 'DF2': data_img}
   df = pd.concat(dfs_dictionary)
   df = df.drop_duplicates(keep=FALSE)
   return df['ImageId'].tolist()
def extract_data(path):
    if  not os.path.exists(path):
        os.mkdir(path)
        normal= os.path.join(path, 'normal')
        os.mkdir(normal)
        defects = os.path.join(path, 'defect')
        os.mkdir(defects)
def parse_data_from_input(filename,source_dir,target_dir):
    ls = find_files(source_dir, filename)
    for row in ls:
        temp_test_data = source_dir + "/" + row
        ziel_dir = os.path.join(target_dir, 'normal/')
        final_val_data = ziel_dir +  row
        shutil.copyfile(temp_test_data, final_val_data)
    with open(filename,'r') as file:
        csv_reader = csv.reader(file,delimiter=',')
        # Skip header
        next(csv_reader, None)
        for row in csv_reader:
            ziel_dir = os.path.join(target_dir,'defect/')
            temp_test_data = source_dir + "/" + row[0]
            final_val_data = ziel_dir +  row[0]
            shutil.copyfile(temp_test_data, final_val_data)
