import streamlit as st
import environment
import gymnasium as gym
import numpy as np
from ppo import PPOAgent
from sac import SACAgent
from dqn import DQNAgent
from a2c import A2CAgent

environment.init()
env = gym.make('WaterHeater-v0')

st.header("Water Heater RL Approach")
st.divider()
st.write(env.ROOM_TEMP)