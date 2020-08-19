"""
Python file is retrieving all the zip folders present in the directory and
extracting all the files in those zip folders in the current directory
"""
import os, zipfile
from data_preprocessing import dir_name
if __name__ == '__main__':

    extension = ".zip"

    os.chdir(dir_name) # change directory from working dir to dir with files

    for item in os.listdir(dir_name): # loop through items in dir
        if item.endswith(extension):
            try:
                file_name = os.path.abspath(item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall() # extract file to dir
                zip_ref.close()
                os.remove(file_name)# close file
            except Exception as e:
                print(e)