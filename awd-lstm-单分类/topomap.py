# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:54:22 2019

@author: night
"""

from mne.viz import plot_topomap
import numpy as np
import matplotlib.pyplot as plt



class heatmap:
    def __init__(self):
        self.pos =   np.array([[0.09456747, 0.45      ],
                               [0.19883673, 0.6       ],
                               [0.29158346, 0.68      ],
                               [0.37592077, 0.72      ],
                               [0.4617831 , 0.73      ],
                               [0.54879398, 0.72      ],
                               [0.63632404, 0.68      ],
                               [0.72916309, 0.6       ],
                               [0.82952896, 0.45      ],
                               [0.182265  , 0.27      ],
                               [0.282265  , 0.38      ],
                               [0.372265  , 0.42      ],
                               [0.46226497, 0.43      ],
                               [0.552265  , 0.42      ],
                               [0.642265  , 0.38      ],
                               [0.742265  , 0.27      ],
                               [0.31271   , 0.07      ],
                               [0.28271   , 0.2       ],
                               [0.46271014, 0.2       ],
                               [0.64271   , 0.2       ],
                               [0.61271   , 0.07      ]])#channel position




