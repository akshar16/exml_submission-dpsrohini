import socketio
import requests
import os
import time
import numpy as np
import tensorflow as tf
from keras import layers
from collections import deque

class GhostAgent:
    def __init__(self, state_shape=(31, 28, 1), action_space=4):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
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
        state = np.array(board, dtype=str)
    
        state = np.where(state == '#', '1', state)
        state = np.where(state == 'p', '5', state)
        state = np.where(state == '.', '3', state)
        state = np.where(state == ' ', '0', state)
        state = np.where(state == 'o', '4', state)
        state = np.where(state == 'a', '5', state)
        state = np.where(state == 'b', '5', state)
        state = np.where(state == 'c', '5', state)
        state = np.where(state == 'd', '5', state)
    
        # Convert all elements to floats after replacements for model compatibility
        state = state.astype(float) / 5.0
        
        # Reshape state for input to the model
        return state.reshape(1, *self.state_shape)



    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [[np.random.randint(2) for _ in range(4)] for _ in range(4)]
        
        act_values = self.model.predict(state, verbose=0)
        moves = []
        for _ in range(4): 
            move = [0, 0, 0, 0]
            move[np.argmax(act_values[0])] = 1
            moves.append(move)
        return moves

link = "http://127.0.0.1:5000"

token = input('Enter your token: ')
name = input('Enter your user name: ')

ghost_agent = GhostAgent()
try:
    ghost_agent.model = tf.keras.models.load_model('ghost_model.keras')
except:
    print("Starting with new model")

sio = socketio.Client()
connected = False

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
    ghost_agent.model.save('ghost_model.keras')
    os._exit(0)

@sio.on('board')
def handle_server_message(data):
    if not connected:
        print("Not connected yet, ignoring message")
        return
    board, points = data
    process(board, points)

@sio.on('reset')
def reset():
    print("Resetting")
    ghost_agent.model.save('ghost_model.keras')
    os._exit(0)   

def process(board, points):

    state = ghost_agent.process_state(board)
    move = ghost_agent.act(state)
    if np.random.rand() < 0.1: 
        ghost_agent.model.save('ghost_model.keras')
    send_move(move)

def send_move(move):
    url = f"{link}/move/ghost"
    payload = move
    headers = {'content-type': 'application/json', 'Authorization': f'Bearer {token}'}
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)

sio.connect(link, headers={'Authorization': f'Bearer {token}', 'Name': name})
sio.wait()
