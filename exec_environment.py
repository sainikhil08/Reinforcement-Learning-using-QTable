"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""

import sys
# Change to the path of your ZMQ python API
sys.path.append('/app/zmq/')
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import time

class Simulation():
    def __init__(self, sim_port = 23004):
        self.sim_port = sim_port
        self.directions = ['Up','Down','Left','Right']
        self.qtable = np.zeros((1000, len(self.directions)))
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost',port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()
    
    def getObjectHandles(self):
        self.tableHandle=self.sim.getObject('/Table')
        self.boxHandle=self.sim.getObject('/Table/Box')
    
    def dropObjects(self):
        self.blocks = 18
        frictionCube=0.06
        frictionCup=0.8
        blockLength=0.016
        massOfBlock=14.375e-03
        
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
        self.client.step()
        retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue=self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break
    
    def getObjectsInBoxHandles(self):
        self.object_shapes_handles=[]
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step
    
    def action(self,direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir*span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

    def updateQTable(self, state, action, reward, new_state, alpha, gamma):
        # Update the Q-table using the Q-learning formula
        old_value = self.qtable[state, action]
        next_max = np.max(self.qtable[new_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        self.qtable[state, action] = new_value


    def choose_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.directions)
        else:
            action = np.argmax(self.qtable[state,:])
            return self.directions[action]

    def calculate_reward(self, positions):
        blue_positions = positions[:9]
        red_positions = positions[9:]

        blue_centroid = np.mean(blue_positions, axis=0)
        red_centroid = np.mean(red_positions, axis=0)
        distance = np.linalg.norm(blue_centroid - red_centroid)

        reward = max(0, 10 - distance)
        return reward
    
def get_state_index(positions):

    discretized_positions = np.round(positions, decimals=1)
    state_str = ''.join(map(str, discretized_positions.flatten()))
    state_index = hash(state_str) % 1000 

    return state_index


def isSuccess(env):

    positions = env.getObjectsPositions()
    blue_positions = positions[:9]
    red_positions = positions[9:]

    blue_centroid = np.mean(blue_positions, axis=0)
    red_centroid = np.mean(red_positions, axis=0)
    distance = np.linalg.norm(blue_centroid - red_centroid)

    success_threshold = 1.0
    return distance < success_threshold

def train_agent(episodes, steps):
    
    alpha = 0.1
    gamma = 0.9
    epsilon = 1
    f = open('rewards.txt', 'w')
    rewards=[]
    for episode in range(episodes):
        env = Simulation()
        print(f'Running episode: {episode + 1}')
        total_reward = 0
        positions = env.getObjectsPositions()
        state = get_state_index(positions)
        for step in range(steps):
            # direction = np.random.choice(env.directions)
            # print(f'Step: {step+1}|| Action: {direction}')
            action = env.choose_action(state, epsilon)
            env.action(action)
            positions = env.getObjectsPositions()
            new_state = get_state_index(positions)
            reward = env.calculate_reward(positions)
            env.updateQTable(state,env.directions.index(action),reward, new_state, alpha, gamma)
            # blue_objs = positions[:9]
            # red_objs = positions[9:]
            state = new_state
            total_reward += reward
        env.stopSim()
        rewards.append(total_reward)
        f.write(f"Reward for episode {episode+1} : {total_reward}\n")
        epsilon = max(0.1,epsilon*0.995)

    
    np.save("qtable.npy", env.qtable)


def test_agent(env, episodes, use_q_table):

    results=[]
    total_time_taken=0
    for episode in range(episodes):
        start_time = time.time()
        state = get_state_index(env.getObjectsPositions())
        total_reward = 0

        print(f'Running episode: {episode + 1}')
        for step in range(20):
            if use_q_table:
                action = env.choose_action(state, epsilon=0)  # Exploitation
            else:
                action = np.random.choice(env.directions)  # Random Exploration

            env.action(action)  
            # print(f'Step: {step+1}|| Action: {action}')
            new_state = get_state_index(env.getObjectsPositions())
            reward = env.calculate_reward(env.getObjectsPositions())
            total_reward += reward
            state = new_state
        
        end_time = time.time()
        episode_time = end_time - start_time
        total_time_taken += episode_time

        success = isSuccess(env)
        results.append((bool(success), float(total_reward)))
    
    return results, total_time_taken

def main():

    train_agent(episodes=10, steps=50)
    env = Simulation()
    test_results_random, rand_time = test_agent(env, episodes=10, use_q_table=False)
    env.stopSim()
    env = Simulation()
    test_results_q_learning, Q_time = test_agent(env, episodes=10, use_q_table=True)
    env.stopSim()

    with open('test_results.txt', 'w') as f:
        f.write("Results based on Random Actions:\n")
        for result in test_results_random:
            f.write(f"{result}\n")
        f.write(f"\nTotal time taken : {rand_time} seconds")
        f.write("\nResults based on Q-Learning Action:\n")
        for result in test_results_q_learning:
            f.write(f"{result}\n")
        f.write(f"\nTotal time taken : {Q_time} seconds")


if __name__ == '__main__':
    
    main()
