import typing as tp
import cv2
import gymnasium as gym
import imageio
import numpy as np
import plotly.graph_objects as go
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


def estimate_policy(n_experiments: int = 100_000, out_file_name=None, render_every_n_episode: int = 1_000) -> float:
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


def draw_value(title: str, values: tp.Dict[str, np.ndarray]) -> go.Figure:
    fig = make_subplots(rows=1, col=2, shared_xaxes=False, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                        subplot_titles=["No ace", "Ace"])
    fig.add_trace(go.Surface(y=card_names, x=player_scores_axis,
                             z=values['no ace'], colorScale='YlGnBu'), col=1, row=1)
    fig.layout.scene1.camera = camera
    fig.layout.scene1.xaxis.nticks = 9
    fig.layout.scene1.yaxis.nticks = 10
    fig.add_trace(go.Surface(y=card_names, x=player_scores_axis,
                             z=values['ace'], colorscale='YlGnBu'), col=2, row=1)
    fig.layout.scene2.xaxis.nticks = 10
    fig.layout.scene2.yaxis.nticks = 10
    fig.update_layout(scene_camera=camera, title=title,
                      margin=dict(r=25, l=25, b=10, t=80),
                      width=1000,
                      showlegend=False)

