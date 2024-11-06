import socketio
import requests
import time
import os
import numpy as np
import tensorflow as tf
from keras import layers
from collections import deque
import random

link = "http://127.0.0.1:5000"

class PacmanAgent:
    def __init__(self, state_shape=(31, 28, 1)):
        self.state_shape = state_shape
        self.action_space = 4
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model

    def process_state(self, board):
        # Convert board characters to numbers
        state = np.zeros(self.state_shape)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'p':
                    state[i,j] = 0
                elif board[i][j] in 'abcd':
                    state[i,j] = -1
                elif board[i][j] == '.':
                    state[i,j] = 0.75
                elif board[i][j] == '#':
                    state[i,j] = -0.5
        return state.reshape(1, *self.state_shape)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = self.model.predict(states, verbose=0)
        next_targets = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_targets[i])

        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize SocketIO client
token = input('Enter your token: ')
name = input('Enter your user name: ')

sio = socketio.Client()
connected = False
agent = PacmanAgent()

# Try to load pretrained model if it exists
try:
    agent.model = tf.keras.models.load_model('pacman_model.keras')
    print("Loaded pretrained model")
except:
    print("Starting with new model")

@sio.event
def connect():
    global connected
    connected = True
    print("Connected to server")
    sio.emit('request')

@sio.event
def disconnect():
    global connected
    connected = False
    print("Disconnected from server")
    agent.model.save('pacman_model.keras')
    os._exit(0)

def process(board, points):
    state = agent.process_state(board)
    action = agent.act(state)
    
    # Convert action (0,1,2,3) to move format [up,down,left,right]
    move = [0,0,0,0]
    move[action] = 1
    
    send_move(move)
    
    return state, action

def send_move(move):
    url = f"{link}/move/player"
    payload = move
    headers = {'content-type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)

@sio.on('board')
def handle_server_message(data):
    if not connected:
        print("Not connected yet, ignoring message")
        return
    board, points = data
    process(board, points)

sio.connect(link, headers={'Authorization': f'Bearer {token}', 'Name': name})
sio.wait()
