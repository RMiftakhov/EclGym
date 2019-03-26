import sys
import subprocess
import os
import shutil
import glob
from collections import namedtuple
import gym
import numpy as np
from ecl.summary import EclSum
from ecl.eclfile import EclFile
from ecl_gym.envs.read_data_file import read_data

# User sys variables
## Path to a simulator exe file
ECL_EXECUTION_PATH = "path/to/eclipse.exe"
## Path to a folder a simulation model
ECL_SIM_MODEL_PATH = "example/egg_model/"
## The simulation file name 
ECL_SIM_MODEL_NAME = "EGG_MODEL_ECL.DATA"
## Path where to store log files and simulation outputs
LOG_PATH = ECL_SIM_MODEL_PATH + "LOG/"

# Simulation variables
TIME_STEP_SIZE = 1 #days
## Define BHP limit on injection
BHP_INJ_LIM = 300
## Define BHP limit on production
BHP_PROD_LIM = 300
## Define RATE limit on injection
RATE_INJ_LIM = 3000
## Define RATE limit on production
RATE_PROD_LIM = 3000000

# State information related variables
## Half window from the well's position in x direction
OBS_HALF_X = 3
## Half window from the well's position in y direction
OBS_HALF_Y = 3
## Number of cell from first gridblock
OBS_NZ = 3
## State information variables
OBS_RETURN = ["PRESSURE", "SWAT"]

WellStruct = namedtuple("WellStruct", "name i_indx j_indx is_producer")
## The class that defines methods
#  for working with the environment. 
class EclEnv(gym.Env):
    ## The constructor.
    def __init__(self):
        # Create a logging directory
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        else:
            shutil.rmtree(LOG_PATH)
            os.makedirs(LOG_PATH)
        self.simulation_iter = 1
        self.restart_file = ECL_SIM_MODEL_PATH + ECL_SIM_MODEL_NAME.replace(".DATA", "")
        self.file = open(LOG_PATH + "logging.txt", "a+")
        self.wells = WellStruct
        self.wells.name = [] 
        self.wells.i_indx = [] 
        self.wells.j_indx = [] 
        self.wells.is_producer = []
        self.dims, self.solution_kw, self.summary_kw, welspecs = \
        read_data(ECL_SIM_MODEL_PATH + ECL_SIM_MODEL_NAME)
        try:
            [[self.wells.name.append(welspecs[w][0]), self.wells.i_indx.append(int(welspecs[w][2]) - 1), \
            self.wells.j_indx.append(int(welspecs[w][3]) - 1), self.wells.is_producer.append(None)] \
                for w in range(len(welspecs))]
        except ValueError:
            print("### Error in WELSPECS: make sure to follow the format -> 'INJECT1' '1'   5    57  1* 'WATER' /")
            sys.exit(1)
        self.n_agents = len(self.wells.name)
        self.obs_return = OBS_RETURN
        self.glob_to_loc = None
        ## action space for well is 
        #  value, is_rate, is_producer and is_open
        self.action_space = (self.n_agents, 4)
        ## observation space for each agent
        #  defined in the vicinity of an agent
        self.observation_space = ((OBS_HALF_X * 2 + 1), \
            (OBS_HALF_Y * 2 + 1), (OBS_NZ))
        self.obs = np.zeros((self.n_agents, len(self.obs_return), \
            self.observation_space[0]*self.observation_space[1]*self.observation_space[2]))
        self.obs_full = None
        print("Number of agents: ", self.n_agents)
        print("Grid Dimenstions: ", self.dims)

    ## Method takes vector of actions and passes
    #  it to env
    #  @param self The object pointer.
    #  @param action The action vector for each agent.
    def step(self, action):
        value, is_rate, is_producer, is_open = \
            [[action[w][i] for w in range(self.n_agents)] for i in range(4)]
        obs, reward, done, obs_full = \
             self.__step_spec(value, is_rate, is_producer, is_open, TIME_STEP_SIZE)
        return obs, reward, done, obs_full

    ## Internal implementation of step method
    #  @param self The object pointer.
    #  @param value The incoming value for BHP or RATE.
    #  @param is_rate Defines if a well is RATE conrolled.
    #  @param is_producer Defines if a well is producer.
    #  @param is_open Defines if a well is open.
    #  @param time_step_size Timestep size for this step.
    def __step_spec(self, value, is_rate, is_producer, is_open, time_step_size):
        with open(LOG_PATH + ECL_SIM_MODEL_NAME, "w") as file_obj:
            self.__add_restart_and_load_keyword(file_obj)
            file_obj.write("\nSCHEDULE")
            for w_indx in range(self.n_agents):
                self.wells.is_producer[w_indx] = is_producer[w_indx]
                if is_producer[w_indx]:
                    self.__add_production_well(file_obj, self.wells.name[w_indx], \
                        value[w_indx], is_rate[w_indx], is_open[w_indx])
                else:
                    self.__add_injector_well(file_obj, self.wells.name[w_indx], \
                        value[w_indx], is_rate[w_indx], is_open[w_indx])
            file_obj.write("\nTSTEP")
            file_obj.write("\n{} /".format(time_step_size))
            file_obj.write("\nEND")
        self.__run_simulation(LOG_PATH.replace('/mnt/c', 'C:') + ECL_SIM_MODEL_NAME.replace(".DATA", ""))
        obs, rewards, dones, obs_full = self.__get_data_from_file(LOG_PATH.replace('/', '\\') \
            + ECL_SIM_MODEL_NAME)
        return obs, rewards, dones, obs_full

    ## Here you can modify the reward function,
    #  so the agent will learn something useful
    #  @param self The object pointer.
    #  @param summary_file The libecl object of summary file.
    #  @param well_name The well's name.
    def __reward_function(self, summary_file, well_name): 
        return summary_file.numpy_vector("FOPR")[-1]

    ## Here you can modify the done function.
    #  if returns true, simulation stops.
    #  @param self The object pointer.
    #  @param summary_file The libecl object of summary file.
    #  @param well_name The well's name.
    #  @param is_producer Is a producer well.
    def __done_function(self, summary_file, well_name, is_producer):
        if is_producer is True:
            value = summary_file.numpy_vector("WOPR:"+ well_name.replace("'", ""))
        else:
            value = summary_file.numpy_vector("WWIR:"+ well_name.replace("'", ""))
        return bool(value[-1] == 0.0)

    ## Method that reinitialize the environment.
    #  @param self The object pointer.
    def reset(self):
        self.simulation_iter = 1
        self.restart_file = ECL_SIM_MODEL_PATH + ECL_SIM_MODEL_NAME.replace(".DATA", "")
        self.__run_simulation(ECL_SIM_MODEL_PATH.replace('/mnt/c', 'C:') \
            + ECL_SIM_MODEL_NAME.replace(".DATA", ""))
        obs, _, _, _ = self.__get_data_from_file(ECL_SIM_MODEL_PATH.replace('/', '\\') \
            + ECL_SIM_MODEL_NAME, is_reset=True)
        return obs

    ## Method that provides visualization.
    #  @param self The object pointer.
    def render(self, mode):
        return None

    ## Method makes copy of files to LOG directory,
    #  so that a model can be opened in ResInsight.
    #  @param self The object pointer.
    def close(self):
        for file in glob.glob(ECL_SIM_MODEL_PATH+ECL_SIM_MODEL_NAME.replace(".DATA", "")+"*"):
            # print(file)
            shutil.copy(file, LOG_PATH)
             

    ## Method that reads and manages a simulator's output files.
    #  @param self The object pointer.
    #  @param data_file Path to the simulation data file.
    #  @param is_reset Defines if the method is called from the reset function.
    def __get_data_from_file_old(self, data_file, is_reset=False):
        if self.glob_to_loc is None:
            self.__read_init_file(data_file)
        summary = EclSum(data_file, include_restart=False)
        rst_file = EclFile(data_file.strip(".DATA") + f'.X{self.simulation_iter:04}')
        vector = []
        obs = []
        try:
            vector = [rst_file[kw][-1] for kw in self.obs_return]
        except KeyError as e_exept:
            print("### Error: check SUMMARY section, report was not generated for -> {}".format(e_exept))
            sys.exit(1)
        if len(self.obs_return) > 1:
            full_obs = vector[0]
            for i in range(1, len(self.obs_return)):
                full_obs = np.hstack((full_obs, vector[i]))
            for w_indx in range(self.n_agents):
                obs_local = [self.__cut_data_based_on_index(vector[i], self.wells.i_indx[w_indx], \
                    self.wells.j_indx[w_indx]) for i in range(len(self.obs_return))]
                obs.append(obs_local)
        else:
            full_obs = np.hstack((vector[0]))
        if is_reset is False:
            try:
                rewards = [self.__reward_function(summary, well_name) \
                    for well_name in self.wells.name]
                dones = [self.__done_function(summary, self.wells.name[w_indx], \
                    self.wells.is_producer[w_indx]) for w_indx in range(self.n_agents)]
            except KeyError as e_exept:
                print("### Error: make sure that keywords in reward and done function \
                is defined in data file {}".format(e_exept))
                sys.exit(1)
        else:
            rewards = [[0.0]]
            dones = [[False]]
        return np.asanyarray(obs), np.asanyarray(rewards), np.asanyarray(dones), full_obs

    ## Method that reads and manages a simulator's output files.
    #  @param self The object pointer.
    #  @param data_file Path to the simulation data file.
    #  @param is_reset Defines if the method is called from the reset function.
    def __get_data_from_file(self, data_file, is_reset=False):
        if self.glob_to_loc is None:
            self.__read_init_file(data_file)
        summary = EclSum(data_file, include_restart=False)
        rst_file = EclFile(data_file.replace(".DATA", "") + f'.X{self.simulation_iter:04}')
        try:
            if self.obs_full is None: 
                self.obs_full = np.zeros((len(self.obs_return), len(rst_file[self.obs_return[0]][-1])))
            for indx, kw in enumerate(self.obs_return):
                self.obs_full[indx] = rst_file[kw][-1]
        except KeyError as e_exept:
            print("### Error: check SUMMARY section, report was not generated for -> {}".format(e_exept))
            sys.exit(1)

        for w_indx in range(self.n_agents):
            self.obs[w_indx] = [self.__cut_data_based_on_index(self.obs_full[i], self.wells.i_indx[w_indx], \
                self.wells.j_indx[w_indx]) for i in range(len(self.obs_return))]

        if is_reset is False:
            try:
                rewards = [self.__reward_function(summary, well_name) \
                    for well_name in self.wells.name]
                dones = [self.__done_function(summary, self.wells.name[w_indx], \
                    self.wells.is_producer[w_indx]) for w_indx in range(self.n_agents)]
            except KeyError as e_exept:
                print("### Error: make sure that keywords in reward and done function \
                is defined in data file {}".format(e_exept))
                sys.exit(1)
        else:
            rewards = [[0.0]]
            dones = [[False]]
        return self.obs, np.asanyarray(rewards), np.asanyarray(dones), self.obs_full

    ## Method that returns data based on
    #  a specified 3D window.
    #  @param self The object pointer.
    #  @param datavec Full observation vector.
    #  @param i_indx Starting position for x axis.
    #  @param j_indx Starting position for y axis.
    def __cut_data_based_on_index(self, datavec, i_indx, j_indx):
        well_data = []
        for i in range(i_indx - OBS_HALF_X, i_indx + OBS_HALF_X + 1):
            for j in range(j_indx - OBS_HALF_Y, j_indx + OBS_HALF_Y + 1):
                for k in range(0, OBS_NZ):
                    try:
                        well_data.append(datavec[int(self.glob_to_loc[self.__coord_to_loc(i, j, k)])])
                    except:
                        well_data.append(0)
        return well_data

    ## Method that read EGRID file and
    #  constructs maping from global indexing 
    #  to local based on ACTNUM keyword
    #  @param self The object pointer.
    #  @param data_file Path to the simulation data file.
    def __read_init_file(self, data_file):
        number_of_cells = self.dims[0]*self.dims[1]*self.dims[2]
        self.glob_to_loc = np.array([number_of_cells*2]*number_of_cells, dtype=np.int32)
        try:
            actnum = EclFile(data_file.replace(".DATA",".EGRID"))['ACTNUM'][0]
        except IOError:
            print("### Error! Please generate EGRID data")
            sys.exit(-1)
        counter = 0
        for k in range(self.dims[2]):
            for j in range(self.dims[1]):
                for i in range(self.dims[0]):
                    loc = self.__coord_to_loc(i, j, k)
                    if actnum[loc]:
                        self.glob_to_loc[loc] = counter
                        counter += 1

    ## Method that returns a position on 1D vector 
    #  based on 3D indexing (i,j,k)
    #  @param self The object pointer.
    #  @param i Index in x direction.
    #  @param j Index in y direction.
    #  @param k Index in z direction.
    def __coord_to_loc(self, i, j, k):
        return i + j*self.dims[0] + k*self.dims[0]*self.dims[1]

    ## Method that runs simulation with 
    #  specified datafile.
    #  @param self The object pointer.
    #  @param data_file Path to the simulation data file.
    def __run_simulation(self, data_file):
        subprocess.call([ECL_EXECUTION_PATH, data_file], \
        stdout=self.file, stderr=self.file)

    ## Method that adds WCONPROD keyword into
    #  the datafile with all the settings.
    #  @param self The object pointer.
    #  @param file_obj The opened file object.
    #  @param well_name Name of the well.
    #  @param value The incoming value for BHP or RATE.
    #  @param is_rate Defines if a well is RATE conrolled.
    #  @param is_open Defines if a well is open.
    def __add_production_well(self, file_obj, well_name, value, is_rate, is_open):
        is_open_str = 'SHUT'
        if is_open: 
            is_open_str = 'OPEN'
        file_obj.write("\nWCONPROD")
        if is_rate:
            file_obj.write("\n{} '{}' 'ORAT' {}  4*  {}   /" \
                .format(well_name, is_open_str, value, BHP_PROD_LIM))
        else:
            file_obj.write("\n{} '{}' 'BHP' {}  4*  {}   /" \
                .format(well_name, is_open_str, RATE_PROD_LIM, value))
        file_obj.write("\n/")

    ## Method that adds WCONINJE keyword into
    #  the datafile with all the settings
    #  @param self The object pointer.
    #  @param file_obj The opened file object.
    #  @param well_name Name of the well.
    #  @param value The incoming value for BHP or RATE.
    #  @param is_rate Defines if a well is RATE conrolled.
    #  @param is_open Defines if a well is open.
    def __add_injector_well(self, file_obj, well_name, value, is_rate, is_open):
        is_open_str = 'SHUT'
        if is_open: 
            is_open_str = 'OPEN'
        file_obj.write("\nWCONINJE")
        if is_rate:
            file_obj.write("\n{} 'WATER' '{}' 'RATE'  {}  1*   {}   /" \
                .format(well_name, is_open_str, value, BHP_INJ_LIM))
        else:
            file_obj.write("\n{} 'WATER' '{}' 'BHP'  {}  1*   {}   /" \
                .format(well_name, is_open_str, RATE_INJ_LIM, value))   
        file_obj.write("\n/")

    ## Method that adds LOAR, RESTART,
    #  and SUMMARY keywords into the datafile.
    #  SUMMARY is composed by reading the initial datafile.
    #  @param self The object pointer.
    #  @param file_obj The opened file object.
    def __add_restart_and_load_keyword(self, file_obj):
        file_obj.write("\nLOAD")
        file_obj.write("\n'{}' /".format(ECL_SIM_MODEL_PATH + \
            ECL_SIM_MODEL_NAME.replace(".DATA", "")))
        file_obj.write("\nRESTART")
        file_obj.write("\n'{}' {} /".format(self.restart_file.replace(".DATA", ""), \
            self.simulation_iter))
        file_obj.write("\n")
        self.__add_data(self.solution_kw, file_obj)
        file_obj.write("\nSUMMARY\n")
        self.__add_data(self.summary_kw, file_obj)
        self.restart_file = LOG_PATH + ECL_SIM_MODEL_NAME
        self.simulation_iter += 1

    ## Helper method
    #  @param self The object pointer.
    #  @param container The container.   
    #  @param file_handle The opened file object.
    def __add_data(self, container, file_handle):
        for items in container:
            file_handle.write("\n")
            for item in items:
                file_handle.write(item)
                file_handle.write(" ")

if __name__ == '__main__':
    ECLOBJ = [EclEnv() for i in range(1)]
    [ECLOBJ[i].reset() for i in range(1)]
