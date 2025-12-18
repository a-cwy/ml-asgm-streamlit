import streamlit as st
import environment
import gymnasium as gym
import numpy as np
from ppo import PPOAgent
from sac import SACAgent
from dqn import DQNAgent
from a2c import A2CAgent



st.header("Water Heater RL Approach")
option = st.selectbox("Pick model to use", ("PPO", "SAC", "A2C", "DQN"), index = 0, placeholder = "Select model...")
if st.button("Run"):
    env = environment.WaterHeaterEnv()
    obs, _ = env.reset()
    total_reward = 0.0

    match option:
        case "PPO":
            agent = PPOAgent(4, (6,))
            pass
        case "SAC":
            agent = SACAgent()
            pass
        case "A2C":
            agent = DQNAgent(env, 6, 4)
            agent.load_models('dash/target_dqn.keras')
            pass
        case "DQN":
            agent = A2CAgent(env)
            pass

    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, term, trun, info = env.step(action)
        done = term or trun
        total_reward = total_reward + reward
        obs = next_obs

        st.divider()
        st.write(f"Room Temperature: {env.ROOM_TEMP:.2f}C")
        st.write(f"Water Temperature: {env.water_tank_temp:.2f}C")
        st.write(f"Temperature Change: {env.temp_loss:.2f}C")
        st.divider()
        st.write(f"Comfort Reward: {info["rewards"]["comfort"]:.2f}C")
        st.write(f"Hygiene Reward: {info["rewards"]["hygiene"]:.2f}C")
        st.write(f"Energy Reward: {info["rewards"]["energy"]:.2f}C")
        st.write(f"Safety Reward: {info["rewards"]["safety"]:.2f}C")
        st.write(f"Step Reward: {reward:.2f}C")

    st.write(f"Total Reward for Episode: {total_reward:.2f}")