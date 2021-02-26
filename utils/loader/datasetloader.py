'''
Loading external force and computed displacement from vtk simulation files
'''
import os
import pyvista as pv
import torch
import json
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T

class DatasetLoader:
    
    @staticmethod    
    def load(config):
        '''
        input: data path to vtk simulation files
        return: a list includes path to individual path to each file
        '''
        FORCE_MEAN, FORCE_STD  = DatasetLoader.dic_to_np(config.json_force)
        DISP_MEAN, DISP_STD = DatasetLoader.dic_to_np(config.json_disp)

        print("FORCE_MEAN",FORCE_STD)
        print("DISP_STD",DISP_STD)
 
        dataset = []
        knn = T.KNNGraph(k=6)
        print('[INFO] Loading dataset ...')
        for file_path in sorted(os.listdir(config.data_path)):
            
            # print('[INFO] Reading Folder named: {}'.format(folder))
            if file_path.split('.')[-1] == 'vtk':

                full_path = os.path.join(config.data_path, file_path)

                mesh_pv = pv.read(full_path)
                force = mesh_pv.point_arrays['externalForce']
                disp = mesh_pv.point_arrays['computedDispl']

                force_norm = (force - FORCE_MEAN)/FORCE_STD
                disp_norm =  (disp - DISP_MEAN)/DISP_STD

                point_torch = torch.from_numpy(mesh_pv.points)
                disp_torch = torch.from_numpy(disp_norm) #labels
                force_torch = torch.from_numpy(force_norm) #node features

                data = Data(x=force_torch, y=disp_torch, pos=point_torch)
                
                data = knn(data)
                dataset.append(data)
                

        return dataset

    @staticmethod   
    def dic_to_np(json_file):

        dic = json.loads(open(json_file).read())

        mean_x = float(list(dic.values())[0])
        mean_y = float(list(dic.values())[1])
        mean_z = float(list(dic.values())[2])

        std_x = float(list(dic.values())[3])
        std_y = float(list(dic.values())[4])
        std_z = float(list(dic.values())[5])       
        # print("{}/{}/{}".format(mean_x, mean_y, mean_z))
        mean_array = np.array([mean_x, mean_y, mean_z], dtype=np.float32).reshape(1,3)
        std_array = np.array([std_x, std_y, std_z], dtype=np.float32).reshape(1,3)

        return mean_array, std_array

        

    


    
 