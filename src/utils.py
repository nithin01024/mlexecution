import os 
import sys 
import pandas as pd
import pandas as pd
import dill

from src.exception import CoustomException
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except:
        pass