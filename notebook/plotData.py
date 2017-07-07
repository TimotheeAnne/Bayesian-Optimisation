import pickle
import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
from utils import *

def compare_average( N, ID, names, label_names, colors = ['r','b','g','y'], ylabel = "exploration"):
    linestyle = []
    ax = plt.subplot(111)
    shift = 0.94 - 0.055*((len(names)-1)/2)
    if len(colors)<len(N):
        colors = []
        n = len(N)
        for i in range(n):
            colors.append(utils.rainbow((1.0*i)/n))
    for _ in range(len(N)):
        linestyle.append("-")
    handles = []
    for i in range(len(names)):
        L = [[] for _ in range(N[i])]
        for j in ID[i]:
            f = open("../Data/"+names[i]+str(j))
            l = np.array(pickle.load(f))
            for k in range(min(len(L),len(l))):
                L[k].append(l[k])
        mean = []
        sd = []
        for j in range(len(L)):
            mean.append( np.mean(L[j]))
            sd.append( np.std(L[j]))
        mean = np.array( mean)
        sd = np.array(sd)
        x = [jj for jj in range(len(mean))]
        plt.plot( mean,color = colors[i], linestyle = linestyle[i])
        ax.fill_between(x,mean +sd , mean - sd, facecolor = colors[i], alpha = 0.2)
        plt.plot( mean + sd,color = colors[i], linestyle = linestyle[i], alpha = 0.3 )
        plt.plot( mean - sd ,color = colors[i], linestyle = linestyle[i], alpha = 0.3)
        handles.append( mlines.Line2D([], [],linestyle = linestyle[i], color=colors[i], label=label_names[i]))
    plt.xlabel("iterations")
    plt.ylabel(ylabel)
    plt.title("Comparison of the "+ ylabel +" through the iterations of the learning")
    plt.legend(handles=handles, numpoints = 1,bbox_to_anchor=(0., shift, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.ioff()
    plt.show()

def compareTime( names, ID, label):
    T = []
    for i in range(len(names)):
        t = []
        for j in ID[i]:
            f = open("../Data/"+names[i]+j)
            t.append(pickle.load(f))
        T.append(t)
    plt.boxplot(T)
    plt.title("Time")
    plt.xlabel("Method")
    plt.ylabel("Time (s)")
    plt.yscale('log')
    plt.gca().xaxis.set_ticklabels(label)
    plt.ioff()
    plt.show()

def compareCompetence (namesBasic, namesLCB, ID, label):
    competenceBasic = [[],[]]
    competenceLCB = [[],[]]
    handles = []
    competenceName = ["Competence NN b 1","Competence NN b 20","Competence BO b 1","Competence BO b 20"]
    competence1 = []
    competence20 = []
    for i in range(len(namesBasic)):
        comp1 = []
        comp2 = []
        for j in ID[i]:
            f = open("../Data/"+namesBasic[i]+j)
            comp = pickle.load(f)
            comp1 = comp1 + comp[0]
            comp2 = comp2 + comp[1]
        competence1.append(comp1)
        competence20.append(comp2)
    competenceBasic[0] = competence1
    competenceBasic[1] = competence20
    competence1 = []
    competence20 = []
    for i in range(len(namesLCB)):
        comp1 = []
        comp2 = []
        for j in ID[i]:
            f = open("../Data/"+namesLCB[i]+j)
            comp = pickle.load(f)
            comp1 = comp1 + comp[0]
            comp2 = comp2 + comp[1]
        competence1.append(comp1)
        competence20.append(comp2)
    competenceLCB[0] = competence1
    competenceLCB[1] = competence20
    plt.subplot(211)
    plt.title("Measure of Competence")
    plt.ylabel("Average distance to the goal")
    bp = plt.boxplot(competenceBasic[0] + competenceBasic[1] + competenceLCB[0] + competenceLCB[1] , patch_artist=True)
    plt.gca().xaxis.set_ticklabels(label*4)

    N = len(namesBasic) * 4
    colors = [rainbow((j/len(namesBasic))/ (4.)) for j in range(N)]
    i = -1
    for box in bp['boxes']:
        i+=1
        box.set (color = colors[i], linewidth = 2)
        box.set( facecolor = 'white' )

    plt.subplot(212)
    plt.ylabel("Average distance to the goal")
    plt.xlabel("Type of learning")
    plt.gca().xaxis.set_ticklabels(label*4)
    bp = plt.boxplot(competenceBasic[0] + competenceBasic[1] + competenceLCB[0] + competenceLCB[1] , patch_artist=True)
    i = -1
    for box in bp['boxes']:
        i+=1
        box.set (color = colors[i], linewidth = 2)
        box.set( facecolor = 'white' )
    plt.yscale('log')
    for i in range(4):
        handles.append( mlines.Line2D([], [],linestyle = "-", color=colors[i*len(namesBasic)], label=competenceName[i]))
    plt.gca().xaxis.set_ticklabels([""] * N)
    plt.legend(handles=handles, numpoints = 1,bbox_to_anchor=(0., 0.9, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)

    plt.ioff()
    plt.show()

def compare_AB_Basic(ret = False):
    N = [1000,1000,1000]
    ID = [[str(i) for i in range(20)],[str(i) for i in range(20)],[str(i) for i in range(20)]]
    name = ["_env_AB_longBasic_coef_0.2_iSameGoal_1_i_1000_","_env_AB_longBasic_coef_0.2_iSameGoal_10_i_1000_",
            "_env_AB_longBasic_coef_0.2_iSameGoal_30_i_1000_"]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = ["NN eps 0.2 b 1","NN eps 0.2 b 10","NN eps 0.2 b 30"]
    labelComp_names = ["NN\n eps 0.2\n b 1","NN\n eps 0.2\n b 10","NN\n eps 0.2\n b 30"]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compare_average( N, ID, names[1], label_names, ylabel = ylabel[1])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, labelComp_names )

def compare_SA_Basic(ret = False):
    N = [
    #~ 100,100,100,
    #~ 100,100,100,
    100,100,100
    ]
    ID =[
        #~ [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
        #~ [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
        [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
        ]
    name = [
            #~ "_env_SA_longBasic_coef_0.1_iSameGoal_1_i_100_","_env_SA_longBasic_coef_0.5_iSameGoal_1_i_100_",
            #~ "_env_SA_longBasic_coef_1_iSameGoal_1_i_100_",
             #~ "_env_SA_longBasic_coef_0.1_iSameGoal_10_i_100_","_env_SA_longBasic_coef_0.5_iSameGoal_10_i_100_",
            #~ "_env_SA_longBasic_coef_1_iSameGoal_10_i_100_",
             "_env_SA_longBasic_coef_0.1_iSameGoal_30_i_100_","_env_SA_longBasic_coef_0.5_iSameGoal_30_i_100_",
            "_env_SA_longBasic_coef_1_iSameGoal_30_i_100_",
            ]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = [
    #~ "NN eps 0.1 p 1","NN eps 0.5 p 1","NN eps 1 p 1",
    #~ "NN eps 0.1 p 10","NN eps 0.5 p 10","NN eps 1 p 10",
    "NN eps 0.1 p 30","NN eps 0.5 p 30","NN eps 1 p 30",
    ]
    labelComp_names = [
                    #~ "NN\n 0.1\n 1","NN\n 0.5\n 1","NN\n 1\n 1",
                    "NN\n 0.1\n 10","NN\n 0.5\n 10","NN\n 1\n 10"
                    #~ "NN\n 0.1\n 30","NN\n 0.5\n 30","NN\n 1\n 30"
                        ]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, labelComp_names )

def compare_SA_LCB(ret = False):
    N = [
    100,100,100,100,
    100,100,100,100,
    100,100,100,100
    ]
    ID =[
        [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
        #~ [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
        #~ [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(9)],

        ]
    name = [
            "_env_SA_BO_coef_0_k_50_acquisition_LCB_iSameGoal_1_i_100_","_env_SA_BO_coef_0.2_k_50_acquisition_LCB_iSameGoal_1_i_100_",
            "_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_1_i_100_","_env_SA_BO_coef_2_k_50_acquisition_LCB_iSameGoal_1_i_100_",
            #~ "_env_SA_BO_coef_0_k_50_acquisition_LCB_iSameGoal_10_i_100_","_env_SA_BO_coef_0.2_k_50_acquisition_LCB_iSameGoal_10_i_100_",
            #~ "_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_10_i_100_","_env_SA_BO_coef_2_k_50_acquisition_LCB_iSameGoal_10_i_100_",
            #~ "_env_SA_BO_coef_0_k_50_acquisition_LCB_iSameGoal_30_i_100_","_env_SA_BO_coef_0.2_k_50_acquisition_LCB_iSameGoal_30_i_100_",
            #~ "_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_30_i_100_","_env_SA_BO_coef_2_k_50_acquisition_LCB_iSameGoal_30_i_100_"
            ]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = [
    "LCB eps 0 p 1","LCB eps 0.2 p 1","LCB eps 1 p 1","LCB eps 2 p 1",
    #~ "LCB eps 0 p 10","LCB eps 0.2 p 10","LCB eps 1 p 10","LCB eps 2 p 10",
    #~ "LCB eps 0 p 30","LCB eps 0.2 p 30","LCB eps 1 p 30","LCB eps 2 p 30"
    ]
    labelComp_names = [
                    "LCB\n 0\n 1","LCB\n 0.2\n 1","LCB\n 1\n 1","LCB\n 2\n 1",
                    #~ "LCB\n 0\n 10","LCB\n 0.2\n 10","LCB\n 1\n 10","LCB\n 2\n 10",
                    #~ "LCB\n 0\n 30","LCB\n 0.2\n 30","LCB\n 1\n 30","LCB\n 2\n 30"
                        ]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, labelComp_names )

def compare_SA(ret = False):
    N = [
    100,
    #~ 100,100,
    100,
    #~ 100,100
    ]
    ID =[
        [str(i) for i in range(10)],
        #~ [str(i) for i in range(10)],[str(i) for i in range(10)],
        [str(i) for i in range(10)],
        #~ [str(i) for i in range(10)],[str(i) for i in range(9)],
        ]
    name = [
            "_env_SA_longBasic_coef_1_iSameGoal_1_i_100_",
            #~ "_env_SA_longBasic_coef_1_iSameGoal_10_i_100_",
            #~ "_env_SA_longBasic_coef_1_iSameGoal_30_i_100_",
            "_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_1_i_100_",
            #~ "_env_SA_BO_coef_2_k_50_acquisition_LCB_iSameGoal_10_i_100_",
            #~ "_env_SA_BO_coef_2_k_50_acquisition_LCB_iSameGoal_30_i_100_"
            ]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = [
                    "NN epsilon 1 p 1",
                    #~ "NN eps 1 p 10","NN eps 1 p 30",
                    "LCB eps 1 p 1",
                    #~ "LCB eps 2 p 10","LCB eps 2 p 30",
    ]
    labelComp_names = [
                    "NN\n epsilon 1\n p 1",
                    #~ "NN\n 1\n 10","NN\n 1\n 30",
                    "LCB\n epsilon 1\n 1",
                    #~ "LCB\n 2\n 10","LCB\n 2\n 30",
                        ]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, labelComp_names )

def compare_SA_BO(ret = False):
    N = [100,100,
    #~ 100,100,100,100,
    100,100,
    100,100]
    ID = [
    [str(i) for i in range(10)],[str(i) for i in range(10)],
    #~ [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
    [str(i) for i in range(10)],[str(i) for i in range(10)],
    [str(i) for i in range(10)],[str(i) for i in range(8)]
    ]
    name = [
            "_env_SA_BO_coef_0_k_50_acquisition_LCB_iSameGoal_1_i_100_","_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_1_i_100_",
            #~ "_env_SA_BO_coef_0_k_50_acquisition_LCB_iSameGoal_10_i_100_","_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_10_i_100_",
            #~ "_env_SA_BO_coef_0_k_50_acquisition_LCB_iSameGoal_30_i_100_","_env_SA_BO_coef_1_k_50_acquisition_LCB_iSameGoal_30_i_100_",
            "_env_SA_BO_coef_0_k_50_acquisition_MPI_iSameGoal_1_i_100_","_env_SA_BO_coef_1_k_50_acquisition_MPI_iSameGoal_1_i_100_",
            "_env_SA_BO_coef_0_k_50_acquisition_EI_iSameGoal_1_i_100_","_env_SA_BO_coef_1_k_50_acquisition_EI_iSameGoal_1_i_100_"
            ]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = [
                  "LCB eps0 p1","LCB eps1 p1",
                  #~ "LCB eps0 p10","LCB eps1 p10",
                  #~ "LCB eps0 p30","LCB eps1 p30",
                  "MPI eps0 p1","MPI eps1 p1",
                  "EI eps0 p1","EI eps1 p1"
                  ]
    labelComp_names = [
    "LCB\n eps0\n p1","LCB\n eps1\n p1",
    #~ "LCB\n eps0\n p10","LCB\n eps1\n p10",
    #~ "LCB\n eps0\n p30","LCB\n eps1\n p30",
    "MPI\n eps0\n p1","MPI\n eps1\n p1",
    "EI\n eps0\n p1","EI\n eps1\n p1",
    ]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, labelComp_names)

def compare_AB(ret = False):
    N = [
    1000,
    #~ 1000,1000,1000,1000,
    #~ 1000,1000,
    1000
    ]
    ID = [
    [str(i) for i in range(10)],
    #~ [str(i) for i in range(50)],[str(i) for i in range(50)],[str(i) for i in range(50)],[str(i) for i in range(50)],
    #~ [str(i) for i in range(10)],
    [str(i) for i in range(9)],
    #~ [str(i) for i in range(10)]
    ]
    name = [
    #~ "_env_AB_longBasic_coef_0.001_iSameGoal_1_i_1000_",
     #~ "_env_AB_longBasic_coef_0.01_iSameGoal_1_i_1000_" ,
     "_env_AB_longBasic_coef_0.1_iSameGoal_1_i_1000_",
    #~ "_env_AB_longBasic_coef_0.5_iSameGoal_1_i_1000_","_env_AB_longBasic_coef_1_iSameGoal_1_i_1000_",
    #~ "_env_AB_BO_coef_0_k_50_acquisition_LCB_iSameGoal_1_i_1000_",
    "_env_AB_BO_coef_1_k_50_acquisition_LCB_iSameGoal_1_i_1000_",
    #~ "_env_AB_BO_coef_0.5_k_50_acquisition_LCB_iSameGoal_10_i_1000_"

            ]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = [
    "NN epsilon 0.1 p1",
    #~ "NN epsilon 0.01 p1","NN epsilon 0.1 p1","NN epsilon 0.5 p1","NN epsilon 1 p1",
    #~ "LCB epsilon 0 p1",
    "LCB epsilon 1 p1",
    #~ "LCB epsilon 0.5 p10",
        ]
    label_comp = [
    "NN\n 0.1\n p1",
    #~ "NN\n 0.01\n p1","NN\n 0.1\n p1","NN\n 0.5\n p1","NN\n 1\n p1",
    #~ "LCB\n 0\n p1",
    "LCB\n 1\n p1",
    #~ "LCB\n 0.5\n p1.",
                    ]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compare_average( N, ID, names[1], label_names, ylabel = ylabel[1])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, label_comp )

def compare_AB_BO(ret = False):
    N = [
    1000,
    1000,1000,
    1000,1000,1000,
    1000,1000
    ]
    ID = [
    [str(i) for i in range(10)],
    [str(i) for i in range(10)],[str(i) for i in range(9)],
    [str(i) for i in range(10)],[str(i) for i in range(10)],[str(i) for i in range(10)],
    [str(i) for i in range(10)],[str(i) for i in range(10)],
    ]
    name = [
     "_env_AB_longBasic_coef_0.1_iSameGoal_1_i_1000_",
    "_env_AB_BO_coef_0_k_50_acquisition_LCB_iSameGoal_1_i_1000_","_env_AB_BO_coef_1_k_50_acquisition_LCB_iSameGoal_1_i_1000_",
    "_env_AB_BO_coef_0_k_50_acquisition_LCB_iSameGoal_10_i_1000_","_env_AB_BO_coef_0.1_k_50_acquisition_LCB_iSameGoal_10_i_1000_",
    "_env_AB_BO_coef_1_k_50_acquisition_LCB_iSameGoal_10_i_1000_",
    "_env_AB_BO_coef_0.1_k_50_acquisition_LCB_iSameGoal_30_i_1000_","_env_AB_BO_coef_1_k_50_acquisition_LCB_iSameGoal_30_i_1000_",
            ]
    nameCompBO = ["BO_k_50_iSameGoal_[1, 20]_j_5_"]
    nameCompBasic = ["basic_iSameGoal_[1, 20]_j_20_"]
    exp = [ "exp"+ n for n in name]
    NbB = [ "NbB"+ n for n in name]
    Time = [ "Time"+ n for n in name]
    CompBasic = ["CompBasic"+  n + nameCompBasic[0] for n in name]
    CompBO = ["CompLCB"+  n + nameCompBO[0] for n in name]
    names = [exp,NbB]
    label_names = [
    "NN epsilon 0.1 p1",
    "LCB epsilon 0 p1","LCB epsilon 1 p1",
    "LCB epsilon 0 p10","LCB epsilon 0.1 p10","LCB epsilon 1 p10",
    "LCB epsilon 0.1 p30","LCB epsilon 1 p30",
        ]
    label_comp = [
    "NN\n 0.1\n p1",
    "LCB\n 0\n p1","LCB\n 1\n p1",
    "LCB\n 0\n p10","LCB\n 0.1\n p10","LCB\n 1\n p10",
    "LCB\n 0.1\n p30","LCB\n 1\n p30",
                    ]
    ylabel = ["exploration","number of balls catched","Competence"]
    if ret :
        return N, ID, names, label_names
    compare_average( N, ID, names[0], label_names, ylabel = ylabel[0])
    compare_average( N, ID, names[1], label_names, ylabel = ylabel[1])
    compareTime( Time, ID, label_names)
    compareCompetence( CompBasic,CompBO, ID, label_comp )

#~ compare_Basic_AB()
#~ compare_SA_LCB()
#~ compare_SA()
#~ compare_SA_Basic()
#~ compare_SA_BO()
#~ compare_AB_BO()
compare_AB()
