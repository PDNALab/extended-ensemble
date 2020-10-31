import pyemma
pyemma.__version__
import numpy as np
import glob
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import matplotlib.pyplot as plt
import sys
import shutil
import os
from pyemma.util.contexts import named_temporary_file
from matplotlib.pyplot import cm
from collections import OrderedDict
import mdtraj as md
import itertools
import time
import indices
from indices.base import BaseComparisons as bc
from indices.faith import Faith as Fai


Threshold = 0.6
traj = md.load_dcd('../proteinG_3gb1/yhalfres_0_0/trajectory.00.dcd',top='../proteinG_3gb1/3GB1.pdb')
topfile=traj.top
feat = coor.featurizer(topfile)
residues = np.arange(0,56)
pairs = []                                                                                 
for i,r1 in enumerate(residues):
    for r2 in residues[i+1::2]:
        pairs.append([r1,r2])
pairs = np.array(pairs)
feature=feat.add_residue_mindist(pairs, scheme='closest-heavy',threshold=Threshold,periodic=False)
inp = pyemma.coordinates.load('../proteinG_3gb1/yhalfres_0_0/trajectory.00.dcd', features=feat)