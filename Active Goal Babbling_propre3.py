import matplotlib
matplotlib.use('TkAgg')

from utils import *
from explauto.sensorimotor_model.bayesian_optimisation import BayesianOptimisation
import pickle
from explauto.environment import available_configurations
from explauto import Environment
from time import *

for jj in range(3):
    for i in range(20):
        print "i", i
        # Parameters to change:
        iterations = 100 # Number of iterations
        n_joints = 7      # Number of joints
        n_dmp_basis = 3   # Number of basis per joint
        goal_size = 1.   # Size of the 2D goal space
        sigma_explo_ratio = 0.2 # Exploration noise (standard deviation)
        environmentChoice = ["SA","AB"][0]

        mode = ["basic","longBasic","BO"][2]
        longBasicIterations = [1,10,30]
        acquisition = ['LCB','MPI','EI'][0]
        exploration_weight = [0.2,0.5,1][jj]
        optimisation_iterations = 1
        k = 50

        B = [1,20]
        j1 = 20

        acquisitionCompetence = ['LCB']
        exploration_weightCompetence = 0
        kCompetence = 50
        j2 = 5

        name = "env_" + environmentChoice +"_"
        if mode == "basic":
            name = name + mode +"_coef_" + str(sigma_explo_ratio)
        elif mode == "longBasic":
            name =  name + mode + "_coef_" + str(sigma_explo_ratio) + "_iSameGoal_" + str(longBasicIterations)
        else:
            name = name + mode +"_coef_"+ str(exploration_weight) + "_k_" + str(k) + "_acquisition_" + acquisition +"_iSameGoal_"+ str(optimisation_iterations)

        name = name + "_i_"+ str(iterations)

        print name

        nameCompetence1 = "basic"+ "_iSameGoal_"+ str(B) + "_j_"+ str(j1)
        nameCompetence2 = "BO"+ "_k_" + str(kCompetence) +"_iSameGoal_"+ str(B) + "_j_"+ str(j2)

        do_plot_sensorial_space = 0
        do_plot_measures = 0
        do_save = 1
        do_competence = 1

        explored_s_agb = []
        explorations = []
        number_balls_catched = []
        competence1 = []
        competence2 = []

        if do_plot_sensorial_space:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlim((-1, 1.))
            ax.set_ylim((-1., 1.))
            ax.plot(0., 0., 'sk', ms=8)

        # Definition of the environment
        if environmentChoice == "AB":
            environment = ArmBall(n_joints, n_dmp_basis, goal_size)
        else:
            environment = Environment.from_configuration('simple_arm', 'mid_dimensional')
            environment.noise = 0

        # Initialization of the interest model
        im_model = InterestModel.from_configuration(environment.conf, environment.conf.s_dims, 'random')

        # Initialization of the sensorimotor modelb'
        params = {'acquisition':acquisition, 'exploration_weight' : exploration_weight, 'initial_points': k,'environment' : environment,
                        'optimisation_iterations': optimisation_iterations, 'exact_feval' : True}

        paramsCompetence = []
        for AF in acquisitionCompetence:
            for b in B:
                paramsCompetence.append({'acquisition':AF, 'exploration_weight' : exploration_weightCompetence, 'initial_points': kCompetence,
                'environment' : environment, 'optimisation_iterations': b, 'exact_feval' : True})

        if mode in ["basic","longBasic"]:
            sm_model = SensorimotorModel.from_configuration(environment.conf, 'nearest_neighbor', 'default')
        else :
            sm_model = BayesianOptimisation(environment.conf, **params)

        for _ in range(1):
            # Bootstrap model: 1 random motor babbling
            m = environment.random_motors()[0]
            s = environment.update(m)
            sm_model.update(m,s)


        s_goal = im_model.sample()

        iteration = 0

        TimeAll = 0
        while iteration < iterations:
            iteration = len(explorations)
            if (iteration+1) % 1 == 0:
                clear_output(wait=True)
                print "Iteration:", iteration+1
                if len(number_balls_catched)>0:
                    print "NbC", number_balls_catched[-1]
            time0 = time()
            if mode == "longBasic" and iteration% longBasicIterations == 0:
                s_goal = im_model.sample()
            if random() < 0.2:
                # Do random motor babbling while the ball has not been grasped, and then in 20% of the trials
                m = environment.random_motors()[0]
                s = environment.update(m)
                sm_model.update(m, s)
                if environmentChoice == "SA":
                    explored_s_agb += [s]
                    if do_plot_sensorial_space:
                        ax.add_patch(Circle(tuple(s), 0.05, fc="g", alpha=0.3))
                elif abs(s[-1] - 0.6) > 0.001 : # if the ball has been moved by the arm, we plot its trajectory and last position
                    explored_s_agb += [s] # store s for later evaluation
                    if do_plot_sensorial_space:
                        ax.plot(environment.s_traj[:,0], environment.s_traj[:,1], lw=2, alpha=0.1)
                        ax.add_patch(Circle(tuple(environment.s_traj[-1,:]), 0.1, fc="b", alpha=0.3))

                explorations.append(int(compute_explo(array(explored_s_agb), array([-1., -1.]), array([1., 1.]), gs=grid_size)))
                number_balls_catched.append(len(explored_s_agb))
            else:
                if mode in  ["BO", "basic"] :
                    # Sample a sensory goal maximizing learning progress using the interest model:
                    s_goal = im_model.sample()
                # Infer a motor command to reach that goal using the Nearest Neighbor algorithm:
                m = sm_model.inverse_prediction(tuple(s_goal))
                if  mode in  ["basic","longBasic"]:
                    # Add exploration noise (of variance sigma) to experiment new motor parameters:
                    m = normal(m, sigma_explo_ratio)
                # Execute this command and observe the corresponding sensory effect:
                s = environment.update(m)
                # Update the sensorimotor model:
                sm_model.update(m, s)
                # Update the interest model:
                im_model.update(hstack((m, s_goal)), hstack((m, s)))
                if do_plot_sensorial_space:
                    # Plot the goals in red:
                    ax.plot([s_goal[0]], [s_goal[1]], 'or', alpha=0.3)

                if mode != "BO":
                    if environmentChoice == "SA":
                        explored_s_agb += [s]
                        if do_plot_sensorial_space:
                            ax.add_patch(Circle(tuple(s), 0.05, fc="g", alpha=0.3))
                    elif abs(s[-1] - 0.6) > 0.001 : # if the ball has been moved by the arm, we plot its trajectory and last position
                        explored_s_agb += [s] # store s for later evaluation
                        if do_plot_sensorial_space:
                            ax.plot(environment.s_traj[:,0], environment.s_traj[:,1], lw=2, alpha=0.1)
                            ax.add_patch(Circle(tuple(environment.s_traj[-1,:]), 0.1, fc="b", alpha=0.3))
                    explorations.append(int(compute_explo(array(explored_s_agb), array([-1., -1.]), array([1., 1.]), gs=grid_size)))
                    number_balls_catched.append(len(explored_s_agb))
                else :
                    S = []
                    size = len( sm_model.dataset )
                    for index in range(size - optimisation_iterations, size):
                       S.append(sm_model.dataset.get_y(index))
                    for s in S:
                        if environmentChoice == "SA":
                            explored_s_agb += [s]
                            if do_plot_sensorial_space:
                                ax.add_patch(Circle(tuple(s), 0.02, fc="b", alpha=0.3))
                        elif abs(s[-1] - 0.6) > 0.001: # if the ball has been moved by the arm, we plot its trajectory and last position
                            print s
                            explored_s_agb += [s] # store s for later evaluation
                            if do_plot_sensorial_space:
                                ax.plot(environment.s_traj[:,0], environment.s_traj[:,1], lw=2, alpha=0.1)
                                ax.add_patch(Circle(tuple(environment.s_traj[-1,:]), 0.1, fc="b", alpha=0.3))
                        explorations.append(int(compute_explo(array(explored_s_agb), array([-1., -1.]), array([1., 1.]), gs=grid_size)))
                        number_balls_catched.append(len(explored_s_agb))

            time1 = time()
            TimeAll += time1 - time0

        explorations = explorations[:iterations]
        number_balls_catched = number_balls_catched[:iterations]

        if do_competence:
            if mode in ["basic","longBasic"]:
                dataset = sm_model.model.imodel.fmodel.dataset
            else:
                dataset = sm_model.dataset
            competenceBasic = []
            for b in B:
                competenceBasic.append(mesure_competence(dataset,"NN",environment,j1, b))
            competenceLCB = []
            competenceMPI = []
            competenceEI = []

            for i2 in range(len(B)):
                print "b", [1,20][i2]
                print "LCB"
                #~ time1 = time()
                competenceLCB.append(mesure_competence(dataset,paramsCompetence[i2],environment,j2, 1))
                #~ print "MPI"
                #~ time2 = time()
                #~ print time2-time1
                #~ competenceMPI.append(mesure_competence(dataset,paramsCompetence[3 + i2],environment,j2, 1))
                #~ print "EI"
                #~ time3 = time()
                #~ print time3-time2
                #~ competenceEI.append(mesure_competence(dataset,paramsCompetence[6 + i2],environment,j2, 1))
                #~ time4 = time()
                #~ print time4-time3

        if do_plot_sensorial_space:
            plt.xticks(linspace(-1., 1., 5))
            plt.yticks(linspace(-1., 1., 5))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.grid()
            plt.ioff()
            plt.show()

        Nb_balls = len(explored_s_agb)
        exploration = int(compute_explo(array(explored_s_agb), array([-1., -1.]), array([1., 1.]), gs=grid_size))
        print "Number of ball catch:", Nb_balls
        print "Measure of exploration:", exploration
        print "Time", TimeAll

        if do_plot_measures:
            x  = [j for j in range(len(explorations))]
            plt.subplot(311)
            plt.plot(x, explorations,'g')
            plt.xlabel("number of iterations")
            plt.title("exploration")

            plt.subplot(312)
            plt.plot(x, number_balls_catched,'r')
            plt.xlabel("number of iterations")
            plt.title("number of balls catched")

            if do_competence:
                plt.subplot(313)
                plt.boxplot( competenceBasic )
                #~ plt.boxplot( competenceLCB)
                plt.xlabel("number of iterations")
                plt.title("measure of competence ")
            plt.ioff()
            plt.show()


        if do_save:
            f = open("../saves3/exp_"+ name+"_"+str(i), 'w')
            pickle.dump(explorations, f)
            f.close()
            f = open("../saves3/NbB_"+ name+"_"+str(i), 'w')
            pickle.dump(number_balls_catched, f)
            f.close()
            if do_competence:
                f = open("../saves3/CompBasic_"+ name+ "_" +nameCompetence1 + "_"+str(i), 'w')
                pickle.dump(competenceBasic, f)
                f.close()
                f = open("../saves3/CompLCB_"+ name+ "_" +nameCompetence2 + "_"+str(i), 'w')
                pickle.dump(competenceLCB, f)
                f.close()
            #~ f = open("../saves3/CompMPI_"+ name+ "_" +nameCompetence2 + "_"+str(i), 'w')
            #~ pickle.dump(competenceMPI, f)
            #~ f.close()
            #~ f = open("../saves3/CompEI_"+ name+ "_" +nameCompetence2 + "_"+str(i), 'w')
            #~ pickle.dump(competenceEI, f)
            #~ f.close()
            f = open("../saves3/Time_"+ name+"_"+str(i), 'w')
            pickle.dump(TimeAll, f)
            f.close()


