import shutil
import typing as tp
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm

cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
card_names = ['A', '2', '3', '5', '6', '7', '8', '9', '10']

player_scores_axis = np.arange(12, 22, 1)
dealer_card_offset = 1
player_scores_offset = 12


def default_player_policy(dealer_card: int, player_scores: int, player_has_ace: bool) -> int:
    if player_has_ace:
        if player_scores < 20:
            return 1
        else:
            if player_scores < 16:
                return 1
    return 0


policy_grids: tp.Dict[str, np.ndarray] = {
    'no_ace': np.zeros((10, 10), dtype=int),
    'ace': np.zeros((10, 10), dtype=int)
}

for dealer_card in range(1, 10):
    for player_scores in range(12, 22):
        cell_idx = (dealer_card - dealer_card_offset, player_scores - player_scores_offset)
        policy_grids['no_ace'][cell_idx] = default_player_policy(dealer_card, player_scores, False)
        policy_grids['ace'][cell_idx] = default_player_policy(dealer_card, player_scores, True)


def take_action_from_policy_grid(dealer_card: int, player_scores: int, player_has_ace: bool) -> int:
    global policy_grids
    if player_scores < player_scores_offset:
        return 1
    cell_idx = (dealer_card - dealer_card_offset, player_scores - player_scores_offset)
    assert cell_idx[0] >= 0 and cell_idx[1] >= 0
    action = policy_grids['ace' if player_has_ace else 'no_ace'][cell_idx]
    return action


def estimate_policy(n_experiments: int = 100000, out_file_name=None, render_every_n_episode: int = 1000) -> float:
    global current_avg_reward
    env = gym.make("Blackjack-v1", sab=True, render_mode='rgb_array' if out_file_name is not None else None)

    frames = []
    sum_reward = 0
    for ep_idx in tqdm(range(n_experiments)):
        if out_file_name is not None and ep_idx % render_every_n_episode == 0:
            render_flag = True
            current_avg_reward = (sum_reward / ep_idx) if ep_idx > 0 else 0
        else:
            render_flag = False

        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            player_scores, dealer_card, player_has_ace = obs

            action = take_action_from_policy_grid(dealer_card, player_scores, player_has_ace)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            done = terminated or truncated
            obs = next_obs

            if render_flag:
                frame = env.render()
                cv2.putText(frame, f'Mean Reward: {current_avg_reward:.4f}', (300, 40),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 0), 2, 2)
                frames.append(frame)
        sum_reward += ep_reward

    if out_file_name is not None:
        imageio.v2.mimsave(out_file_name, frames)

    return sum_reward / n_experiments


default_player_reward = estimate_policy(out_file_name="default_policy_game.mp4")
print(default_player_reward)

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.5, y=1.5, z=1.5)
)


def draw_value_function(title: str, values: tp.Dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        shared_xaxes=False,
                        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                        subplot_titles=["No ace", "Ace"])

    fig.add_trace(go.Surface(y=card_names, x=player_scores_axis,
                             z=values['no_ace'], colorscale='YlGnBu'), col=1, row=1)
    fig.layout.scene1.camera = camera
    fig.layout.scene1.xaxis.nticks = 9
    fig.layout.scene1.yaxis.nticks = 10
    fig.add_trace(go.Surface(y=card_names, x=player_scores_axis, z=values['ace'], colorscale='YlGnBu'), col=2, row=1)
    fig.layout.scene2.camera = camera
    fig.layout.scene2.xaxis.nticks = 10
    fig.layout.scene2.yaxis.nticks = 10
    fig.update_layout(scene_camera=camera, title=title,
                      margin=dict(r=25, l=25, b=10, t=80),
                      width=1000,
                      showlegend=False)
    fig.update_scenes(xaxis_title_text='Player',
                      yaxis_title_text='Dealer',
                      zaxis_title_text='Reward')
    return fig


def estimate_value_function(num_episodes: int = 300_000, frame_step: int = 3_000):
    env = gym.make("Blackjack-v1", sab=True)

    temp_dir = Path('tmp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    value_grid_count: tp.Dict[str, np.ndarray] = {
        'no_ace': np.zeros((10, 10), dtype=int),
        'ace': np.zeros((10, 10), dtype=int)
    }
    value_grid: tp.Dict[str, np.ndarray] = {
        'no_ace': np.zeros((10, 10), dtype=float),
        'ace': np.zeros((10, 10), dtype=float)
    }

    value_grid['no_ace'][:, -1] = 1
    value_grid['ace'][:, -1] = 1

    image_paths = []

    for ep_idx in tqdm(range(num_episodes + frame_step)):

        obs, info = env.reset()
        done = False

        while not done:
            player_scores, dealer_card, player_has_ace = obs
            action = take_action_from_policy_grid(dealer_card, player_scores, player_has_ace)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            if player_scores >= player_scores_offset:

                v_g_count = value_grid_count['ace' if player_has_ace else 'no_ace']
                v_g = value_grid['ace' if player_has_ace else 'no_ace']

                cell_idx = (dealer_card - dealer_card_offset, player_scores - player_scores_offset)

                N = v_g_count[cell_idx] + 1
                v_g_count[cell_idx] = N
                if done:
                    td_target = reward
                else:
                    next_player_scores, next_dealer_card, next_player_has_ace = next_obs
                    next_v_g = value_grid['ace' if next_player_has_ace else 'no_ace']

                    next_cell_idx = (next_dealer_card - dealer_card_offset, next_player_scores - player_scores_offset)
                    td_target = reward + next_v_g[next_cell_idx]
                v_g[cell_idx] += (td_target - v_g[cell_idx]) / N

            obs = next_obs

        if ep_idx % frame_step == 0:
            ep_name = str(ep_idx) if ep_idx > 0 else '1'
            fig = draw_value_function(f'Value function on step: {ep_name}', value_grid)

            image_path = temp_dir / f'{ep_name}.png'
            pio.write_image(fig, image_path, format='png')
            image_paths.append(imageio.v2.imread(image_path))

    imageio.v2.mimsave('blackjack_default_value_function.mp4', image_paths, fps=20)


estimate_value_function()
