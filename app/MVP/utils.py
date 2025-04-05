import os
import pickle
import sys
import threading
import tkinter as tk
from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from ultralytics import settings

path_current = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.split(path_current)[0])

from app_sys import AppSys

app_sys = AppSys()

lock = threading.Lock()

def get_log_message(log_type, content):
    """
    log_type: func, var
    """
    return f'{log_type}: {content}'

def direc_exist_check(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_yolo_config(direc):
    """
    Initialise the yolo config, e.g. the default directory of the runs
    """
    # Update a setting
    settings.update({'runs_dir': os.path.join(app_sys.PATH_MODELS, f'{direc}')})
    # Reset settings to default values
    settings.reset()

@dataclass
class VideoData:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path)
        self.__post_init__()

    def __post_init__(self):
        with lock:
            self.vcap = cv2.VideoCapture(self.path)
            self.fps = self.vcap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.sframe = 0
            self.eframe = self.total_frames - 1

            self.vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.vcap.read()
            self.height, self.width, _ = frame.shape

    def set_start_frame(self, frame_number: int):
        with lock:
            # main_logger.info(get_log_message('func', 'VideoData.set_start_frame'))
            self.vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.vcap.read()
            return ret, frame

    def release(self):
        # main_logger.info(get_log_message('func', 'VideoData.release'))
        self.vcap.release()

class AnnotationObjects:
    def __init__(self, img_name, classes:list):
        self.img = img_name
        self.classes = classes
        self.reset_graph()

    def reset_graph(self):
        self.dict_graph = {cls: nx.Graph() for cls in self.classes}

    def add_node(self, cls:str, node:str, coords:tuple):
        # Add node
        self.dict_graph[cls].add_node(node, data=coords)
        # Connect an edge to the previous node
        nodes = list(self.dict_graph[cls].nodes)
        if len(nodes) == 1:
            pass
        else:
            self.add_edge(cls, nodes[-2], nodes[-1])
    
    def add_edge(self, cls:str, node0:str, node1: str):
        self.dict_graph[cls].add_edge(node0, node1)
        self.dict_graph[cls].edges[node0, node1]['tag'] = f'{node0}_{node1}'

    def remove_node(self, cls:str, node:str):
        nodes = list(self.dict_graph[cls].nodes)
        if len(nodes) <= 2:
            self.dict_graph[cls].remove_node(node)
        else:
            ngbrs = self.find_neighbours(cls, node)
            # In case of polygon annotation, only two neighbours exist
            if len(ngbrs) == 2:
                self.add_edge(cls, ngbrs[0], ngbrs[1])
            self.dict_graph[cls].remove_node(node)

    def close_polygon(self, cls:str):
        # Connect the last and the first nodes.
        lis_nodes = self.get_node_coords_all(cls)
        n0_name, n0_data = lis_nodes[0]
        n1_name, n1_data = lis_nodes[-1]
        self.add_edge(cls, n0_name, n1_name)

    def get_node_coords(self, cls:str, node:str):
        return self.dict_graph[cls].nodes[node]['data']

    def get_node_coords_all(self, cls:str):
        return list(self.dict_graph[cls].nodes.data())

    def find_neighbours(self, cls:str, node:str):
        return list(nx.all_neighbors(self.dict_graph[cls], node))

    def find_adj_edges(self, cls:str, node:str):
        """
        {'<node0>': {'tag': '<node0>_<node>'}, '<node1>': {'tag': '<node>_<node1>'}}
        """
        return self.dict_graph[cls].adj[node]

    def update_node(self, cls:str, node:str, coords:tuple):
        # Update node coordinates
        self.dict_graph[cls].nodes[node]['data'] = coords

    def get_all_edges(self, cls):
        return self.dict_graph[cls].edges


"""
Graph representation of the wall keypoints
"""
class WallKeypoints:
    def __init__(self, img_name, **kwargs):
        """
        kwargs: inverty for tkinter visualisation as yaxis inverted in side the canvas.
        """
        self.img = img_name
        # Adjacency matrix file name
        self.path_graph_adj = os.path.join(app_sys.PATH_ASSET, 'kp_adj.csv')
        # Graph object name 
        self.path_graph_init = os.path.join(app_sys.PATH_ASSET, 'kp_graph_init.pkl')

        if not os.path.exists(self.path_graph_init):
            self.init_graph(**kwargs)
        else:
            self.graph = pickle.load(open(self.path_graph_init, 'rb'))

    def init_nodes(self, **kwargs):
        """
        Inititialise the node attributes
        """
        inverty = kwargs.pop('inverty', False)
        self.wall_angle = 40 # degrees
        theta = self.wall_angle / 180 * np.pi

        """
        3D coords
        """
        # X
        xA = -1000
        xC = -600
        xD = -400
        xE = -200
        xF = 0
        xH = 400
        xI = 600
        xK = 1000
        self.coords_3d_x = {
            'xA': xA,
            'xC': xC,
            'xD': xD,
            'xE': xE,
            'xF': xF,
            'xH': xH,
            'xI': xI,
            'xK': xK
        }

        # Y
        y1 = 100 * np.cos(theta) + 370
        y4 = y1 + 200 * np.cos(theta) * 3
        y6 = y1 + 200 * np.cos(theta) * 5
        y8 = y1 + 200 * np.cos(theta) * 7
        y9 = y1 + 200 * np.cos(theta) * 8
        y11 = y1 + 200 * np.cos(theta) * 10
        y12 = y1 + 200 * np.cos(theta) * 11
        y14 = y1 + 200 * np.cos(theta) * 13
        y18 = y1 + 200 * np.cos(theta) * 17
        self.coords_3d_y = {
            'y1': y1,
            'y4': y4,
            'y6': y6,
            'y8': y8,
            'y9': y9,
            'y11': y11,
            'y12': y12,
            'y14': y14,
            'y18': y18,
        }

        # Z
        z1 = 100 * np.sin(theta)
        z4 = z1 + 200 * np.sin(theta) * 3
        z6 = z1 + 200 * np.sin(theta) * 5
        z8 = z1 + 200 * np.sin(theta) * 7
        z9 = z1 + 200 * np.sin(theta) * 8
        z11 = z1 + 200 * np.sin(theta) * 10
        z12 = z1 + 200 * np.sin(theta) * 11
        z14 = z1 + 200 * np.sin(theta) * 13
        z18 = z1 + 200 * np.sin(theta) * 17
        self.coords_3d_z = {
            'z1': z1,
            'z4': z4,
            'z6': z6,
            'z8': z8,
            'z9': z9,
            'z11': z11,
            'z12': z12,
            'z14': z14,
            'z18': z18,
        }

        """
        2D coords
        """
        scale = 15
        xbase = 250
        ybase = 100

        if inverty:
            ybase = -350

        # X
        xA = xbase -5 * scale
        xC = xbase -3 * scale
        xD = xbase -2 * scale
        xE = xbase - scale
        xF = xbase
        xH = xbase + 2 * scale
        xI = xbase + 3 * scale
        xK = xbase + 5 * scale
        self.coords_2d_x = {
            'xA': xA,
            'xC': xC,
            'xD': xD,
            'xE': xE,
            'xF': xF,
            'xH': xH,
            'xI': xI,
            'xK': xK
        }

        # Y
        y1 = ybase
        y4 = y1 + scale * 3
        y6 = y1 + scale * 5
        y8 = y1 + scale * 7
        y9 = y1 + scale * 8
        y11 = y1 + scale * 10
        y12 = y1 + scale * 11
        y14 = y1 + scale * 13
        y18 = y1 + scale * 17
        self.coords_2d_y = {
            'y1': y1,
            'y4': y4,
            'y6': y6,
            'y8': y8,
            'y9': y9,
            'y11': y11,
            'y12': y12,
            'y14': y14,
            'y18': y18,
        }



        x, y, z = 0, 0, 0
        xx, yy = 0, 0
        self.kp = {
            'A1': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'A4': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'A8': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'A11': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'A14': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'A18': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'C4': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'C6': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'C9': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'C12': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'C14': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'C18': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'D1': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'E18': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'F1': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'F4': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'F8': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'F11': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'F14': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'H14': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'H18': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'I1': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'I4': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'I6': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'I9': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'I11': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'K1': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'K4': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'K8': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'K11': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'K14': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
            'K18': {'coords_3d': (x,y,z), 'coords_2d': (xx,yy), 'visibility': 'visible'},
        }
        if not inverty:
            for k, v in self.kp.items():
                col, row = k[0], k[1:]
                v['coords_3d'] = (self.coords_3d_x['x' + col], self.coords_3d_y['y' + row], self.coords_3d_z['z' + row])
                v['coords_2d'] = (self.coords_2d_x['x' + col], self.coords_2d_y['y' + row])
                print(v)
        else:
            for k, v in self.kp.items():
                col, row = k[0], k[1:]
                v['coords_3d'] = (self.coords_3d_x['x' + col], -1 * self.coords_3d_y['y' + row], self.coords_3d_z['z' + row])
                v['coords_2d'] = (self.coords_2d_x['x' + col], -1 * self.coords_2d_y['y' + row])
                print(v)

    def init_edges(self):
        """
        Initialise the edges
        """
        self.kp_connection = {
            'A1': ['A4', 'C4', 'D1'],
            'A4': ['A1', 'C4', 'C6', 'C9', 'A8', 'D1'],
            'A8': ['A4', 'C6', 'C9', 'C12', 'A11'],
            'A11': ['A8', 'C9', 'C12', 'C14', 'A14'],
            'A14': ['A11', 'C12', 'C14', 'C18', 'A18'],
            'A18': ['A14', 'C14', 'C18'],
            'C4': ['D1', 'A1', 'A4', 'C6', 'F8', 'F4', 'F1'],
            'C6': ['C4', 'A4', 'A8', 'C9', 'F8', 'F4'],
            'C9': ['C6', 'A4', 'A8', 'A11', 'C12', 'F11', 'F8'],
            'C12': ['C9', 'A8', 'A11', 'A14', 'C14', 'F14', 'F11', 'F8'],
            'C14': ['C12', 'A11', 'A14', 'A18', 'C18', 'E18', 'F14', 'F11'],
            'C18': ['C14', 'A14', 'A18', 'E18', 'F14'],
            'D1': ['A1', 'A4', 'C4', 'F4', 'F1'],
            'E18': ['F14', 'C14', 'C18', 'H18', 'H14'],
            'F1': ['D1', 'C4', 'F4', 'I4', 'I1'],
            'F4': ['F1', 'D1', 'C4', 'C6', 'F8', 'I6', 'I4', 'I1'],
            'F8': ['I4', 'C4', 'C6', 'C9', 'C12', 'F11', 'I11', 'I9', 'I6', 'I4'],
            'F11': ['F8', 'C9', 'C12', 'C14', 'F14', 'H14', 'I11', 'I9'],
            'F14': ['F11', 'C12', 'C14', 'C18', 'E18', 'H18', 'H14', 'I11'],
            'H14': ['I11', 'F11', 'F14', 'E18', 'H18', 'K18', 'K14', 'K11'],
            'H18': ['H14', 'F14', 'E18', 'K18', 'K14'],
            'I1': ['F1', 'F4', 'I4', 'K4', 'K1'],
            'I4': ['I1', 'F1', 'F4', 'F8', 'I6', 'K4', 'K1'],
            'I6': ['I4', 'F4', 'F8', 'I9', 'K8', 'K4'],
            'I9': ['I6', 'F8', 'F11', 'I11', 'K11', 'K8', 'K4'],
            'I11': ['I9', 'F8', 'F11', 'F14', 'H14', 'K14', 'K11', 'K8'],
            'K1': ['I1', 'I4', 'K4'],
            'K4': ['K1', 'I1', 'I4', 'I6', 'K8'],
            'K8': ['K4', 'I6', 'I9', 'K11'],
            'K11': ['K8', 'I9', 'I11', 'H14', 'K14'],
            'K14': ['K11', 'I11', 'H14', 'H18', 'K18'],
            'K18': ['K14', 'H14', 'H18'],
        }

    def init_graph(self, **kwargs):
        self.init_nodes(**kwargs)
        self.init_edges()
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(k, v) for k, v in self.kp.items()])
        for k_fr in self.kp.keys():
            if len(self.kp_connection[k_fr]) != 0:
                for k_to in self.kp_connection[k_fr]:
                    self.graph.add_edge(k_fr, k_to, tag=f'{k_fr}_{k_to}')
        
        # Save the graph adjacency matrix as csv file
        df = nx.to_pandas_adjacency(self.graph, nodelist=list(self.kp.keys()), dtype=int)
        df.to_csv(self.path_graph_adj, index=False,header=True)

        # Save the graph object
        pickle.dump(self.graph, open(self.path_graph_init, 'wb'))
    
    def get_all_edges(self):
        return self.graph.edges
    
    def get_node_names(self):
        return [k for k, v in self.get_node_coords_all()]

    def get_node_coords(self, node:str):
        return self.graph.nodes[node]['coords_2d']

    def get_node_coords_all(self):
        return list(self.graph.nodes.data())
    
    def get_node_coords_all_3d(self):
        dict_coords = {}
        for node, data in self.get_node_coords_all():
            dict_coords[node] = data['coords_3d']
        return dict_coords
    
    def update_node(self, node:str, coords:tuple):
        # Update node coordinates
        self.graph.nodes[node]['coords_2d'] = coords

    def find_adj_edges(self, node:str):
        """
        {'<node0>': {'tag': '<node0>_<node>'}, '<node1>': {'tag': '<node>_<node1>'}}
        """
        return self.graph.adj[node]
    
    def find_adj_edges_tags(self, node):
        dict_adj = self.find_adj_edges(node)
        return [v['tag'] for _, v in dict_adj.items()]
    
    def find_neighbours(self, node:str):
        return list(nx.all_neighbors(self.graph, node))

    def update_node_coords_2d(self, node:str, coords:tuple):
        # Update node coordinates
        self.graph.nodes[node]['coords_2d'] = coords

    def reset_graph(self):
        # self.init_graph()
        self.__init__(self.img)


class Holds:
    """
    Get coordinates of all the holds projected to the video frame from the rotayion & translation matrix and the 3d coordinates of the holds
    coords_world: 3d coordinates of all the holds used for solving PnP
    rvec: Rotation vector
    tvec: Translation vector
    K: Camera matrix
    dist: Distortion coefficients
    """
    def __init__(self, **kwargs):
        # wold coordinates of the holds
        self.path_coords_holds = os.path.join(app_sys.PATH_ASSET, 'world_coords_holds.pkl')

        if not os.path.exists(self.path_coords_holds):
            self.init_nodes(**kwargs)
        else:
            self.coords_world = pickle.load(open(self.path_coords_holds, 'rb'))
        self.rvec = kwargs.pop('rvec', None)
        self.tvec = kwargs.pop('tvec', None)
        self.K = kwargs.pop('K', None)
        self.dist = kwargs.pop('dist', None)

    def init_nodes(self, **kwargs):
        """
        Inititialise the node attributes
        """
        self.coords_world = {}
        inverty = kwargs.pop('inverty', False)
        self.wall_angle = 40 # degrees
        theta = self.wall_angle / 180 * np.pi

        cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        y0 = 100 * np.cos(theta) + 370
        z0 = 100 * np.sin(theta)
        for col in cols:
            for row in range(1,19):
                # Centred on column F
                xx = (cols.index(col) - 5) * 200
                yy = y0 + 200 * np.cos(theta) * (row - 1)
                zz = z0 + 200 * np.sin(theta) * (row - 1)
                if inverty:
                    yy = -1 * yy
                self.coords_world[col + str(row)] = (xx, yy, zz)

        pickle.dump(self.coords_world, open(self.path_coords_holds, 'wb'))

    def get_projection(self):
        """
        Get the projection of the holds world coordinates to the video perspective
        """
        # Map the 3D point to 2D point
        world_coords = np.array(list(self.coords_world.values())).astype('float32')
        coords_2d, _ = cv2.projectPoints(world_coords, 
                                        self.rvec, self.tvec, 
                                        self.K, 
                                        self.dist)
        self.coords_frame = {k: v for k, v in zip(self.coords_world.keys(), coords_2d)}
        return self.coords_frame
