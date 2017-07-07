import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
from numpy import pi, array, linspace, hstack, zeros, transpose
from matplotlib import animation
from IPython.display import HTML, display, Image, clear_output
from ipywidgets import interact_manual
from numpy.random import random, normal, uniform
import math


from explauto import SensorimotorModel
from explauto.sensorimotor_model.non_parametric import NonParametric
from explauto import InterestModel
from explauto.interest_model.discrete_progress import DiscretizedProgress
from explauto.utils import rand_bounds, bounds_min_max, softmax_choice, prop_choice
from explauto.environment.dynamic_environment import DynamicEnvironment
from explauto.interest_model.competences import competence_exp, competence_dist
from explauto.environment.modular_environment import FlatEnvironment, HierarchicalEnvironment

from environment import Arm, Ball,ArmBall
from explauto.sensorimotor_model.bayesian_optimisation import BayesianOptimisation
grid_size = 10

import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.lines as mlines

def plot_motor_space(X):
            def cut(l, color = "blue", marker = "o" , ms = 5, title = "Succesive position in the motor space" , x_lab = "x", y_lab ="y"):
                l1 = []
                l2 = []
                n = len(l)
                i = 0.
                colors = ["cyan","blue", "green", "yellow", "red", "magenta"]
                for [x1,x2] in l:
                    if marker == "o":
                        color = rainbow(i/n*0.8)
                        plt.xlabel(x_lab)
                        plt.ylabel(y_lab)
                        plt.title(title)
                    if title == "Two motors dimensions":
                        plt.axis([-np.pi/2.9,np.pi/2.9,-np.pi/2.9,np.pi/2.9])
                    plt.plot(x1,x2,marker =marker, color = color, ms=ms, alpha = 0.5)
                    i+=1.

            def cut_M(X):
                n = len(X[0])/2
                list_m = [ [] for _ in range(n)]
                for x in X:
                    for i in range(n):
                        list_m[i].append([x[2*i],x[2*i+1]])
                return list_m

            M = cut_M(X)
            fig, ax = plt.subplots(figsize=(20,20))
            ax2 = fig.add_axes([0.05, 0.45, 0.9, 0.02])
            N = 10
            cmap = mpl.colors.ListedColormap([rainbow(i/(1.*(N+1))*0.8) for i in range(N+2)])
            cmap.set_over('0.25')
            cmap.set_under('0.75')

            bounds = [int(1.*i/N * len(X)) for i in range(N+1)]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                            norm=norm,
                                            ticks=bounds,  # optional
                                            spacing='proportional',
                                            orientation='horizontal')
            cb2.set_label('Iterations')
            plt.subplot(231)
            cut(M[0], title = "Two motors dimensions", x_lab = "d1", y_lab ="d2" )
            plt.subplot(232)
            cut(M[1],title = "Two motors dimensions",x_lab = "d3", y_lab ="d4" )
            plt.subplot(233)
            cut(M[2],title = "Two motors dimensions",x_lab = "d5", y_lab ="d6" )

def compute_explo(data, mins, maxs, gs=100):
    n = len(mins)
    if len(data) == 0:
        return 0
    else:
        assert len(data[0]) == n
        epss = (maxs - mins) / gs
        grid = np.zeros([gs] * n)
        for i in range(len(data)):
            idxs = np.array((data[i] - mins) / epss, dtype=int)
            idxs[idxs>=gs] = gs-1
            idxs[idxs<0] = 0
            grid[tuple(idxs)] = grid[tuple(idxs)] + 1
        grid[grid > 1] = 1
        return np.sum(grid)

def rainbow(x):
      n = int(x*255*6)
      if (n<=255):
           ret = (255,n,1)
      elif (n<=255*2):
           ret = (255-(n-255),255,1)
      elif (n<=255*3):
           ret = (1,255,n-255*2)
      elif (n<=255*4):
           ret = (1,255-(n-255*3),255)
      elif (n<=255*5):
           ret = (n-255*4,1,255)
      else :
           ret = (255,1,255-(n-255*5))
      (a,b,c) = ret
      return (a/255.,b/255.,c/255.)

def do_min( l ):
    l_min = [l[0]]
    for i in range(1,len(l)):
        l_min.append(min(l_min[-1],l[i]))
    return l_min

def tirage_disque():
    """ http://www.afapl.asso.fr/Tiralea.htm """
    x0 = [0,0]
    R = 1
    u = random()
    radius = R * math.sqrt(u)
    theta = uniform(0., 2 * math.pi)
    x = x0[0] + radius * math.cos(theta)
    y = x0[1] + radius * math.sin(theta)
    res = [x, y]
    return res

def mesure_competence( dataset, methodTest, environment, j, b):
    dist = []
    if methodTest == "NN":
        smTest = SensorimotorModel.from_configuration(environment.conf, 'nearest_neighbor', 'default')
        smTest.model.imodel.fmodel.dataset = dataset
        smTest.t = len(dataset)
        smTest.bootstrapped_s = True
        for _ in range(j):
            s_goal = tirage_disque()
            for _ in range(b-1):
                m = smTest.inverse_prediction(tuple(s_goal))
                s = environment.update(m)
                smTest.update(m,s)
            smTest.mode = "exploit"
            m = smTest.inverse_prediction(tuple(s_goal))
            s = environment.update(m)
            dist.append(np.linalg.norm(s-s_goal))
    else :
        params = methodTest
        smTest = BayesianOptimisation(environment.conf, **params)
        smTest.dataset = dataset
        for _ in range(j):
            s_goal = tirage_disque()
            m = smTest.inverse_prediction(tuple(s_goal))
            s = environment.update(m)
            dist.append(np.linalg.norm(s-s_goal))
    return dist


