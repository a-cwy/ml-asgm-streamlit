import streamlit as st
import matplotlib.pyplot as plt
import environment
import gymnasium as gym
import numpy as np
from ppo import PPOAgent
from sac import SACAgent
from dqn import DQNAgent
from a2c import A2CAgent



st.header("Water Heater RL Approach")
option = st.selectbox("Pick model to use", ("PPO", "SAC", "A2C", "DQN"), index = 0, placeholder = "Select model...")
# season = st.selectbox("Pick season", ("Spring", "Summer", "Autumn", "Winter"), index = 0, placeholder = "Select model...")
n_steps = st.number_input("Number of steps to run", min_value = 1, max_value = 5000, step = 1)
if st.button("Run"):
    env = environment.WaterHeaterEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    reward_hist = []

    match option:
        case "PPO":
            agent = PPOAgent(4, (6,))
            pass
        case "SAC":
            agent = SACAgent()
            pass
        case "DQN":
            agent = DQNAgent(env, 6, 4)
            agent.load_models('dash/target_dqn.keras')
            pass
        case "A2C":
            agent = A2CAgent(env)
            pass

    placeholder = st.empty()

    for step in range(n_steps):
        action = agent.get_action(obs)
        next_obs, reward, _, _, info = env.step(action)
        total_reward = total_reward + reward
        reward_hist.append(reward)
        obs = next_obs

        placeholder.empty()
        with placeholder.container():
            st.divider()
            st.write(f"Day: {env.day:.2f}")
            st.write(f"Time: {env.time:.2f}")
            st.write(f"Room Temperature: {env.ROOM_TEMP:.2f}C")
            st.write(f"Water Temperature: {env.water_tank_temp:.2f}C")
            st.write(f"Temperature Change: {env.temp_loss:.2f}C")
            st.divider()
            st.write(f"Step: {step + 1}")
            st.write(f"Action: {action}")
            st.write(f"Comfort Reward: {info["rewards"]["comfort"]:.2f}")
            st.write(f"Hygiene Reward: {info["rewards"]["hygiene"]:.2f}")
            st.write(f"Energy Reward: {info["rewards"]["energy"]:.2f}")
            st.write(f"Safety Reward: {info["rewards"]["safety"]:.2f}")
            st.write(f"Step Reward: {reward:.2f}")

    st.write(f"Total Reward for Episode: {total_reward:.2f}")
    st.divider()
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(reward_hist))
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Cumulative Reward per Step")
    st.pyplot(fig)