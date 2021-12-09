import pickle
from pandas import read_pickle,DataFrame
from glob import glob


#def load_dc(name_dc):

class Datacube():
    def __init__(self,name_dc):
        self.path_to_datacube = name_dc
        if len(glob(name_dc)) == 1: 
            Mock_dc = read_pickle(name_dc)
        else:
            Mock_dc = DataFrame(data={},columns=['name_id','params','ngal','wp'])
        self.table = Mock_dc
        
    def add_row_dc(self,new_dict):
        Mock_dc = self.table
        self.table = Mock_dc.append(new_dict,ignore_index=True)
        