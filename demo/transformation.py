import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *



Mt = T.translate(1, 0, 0)
Ms = T.scale(1, 2, 1)
Mr = T.rotate(axis=(1., 0., 0.), angle=45.0, radians=None)
print("Mt: "); print(Mt)
print("Ms: "); print(Ms)
print("Mr: "); print(Mr)

