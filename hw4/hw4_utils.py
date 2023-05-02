import logging
import torch
import numpy as np
import gymnasium as gym

logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

try:
    import matplotlib.pyplot as plt
    can_render = True
except:
    logging.warning("Cannot import matplotlib; will not attempt to render")
    can_render = False


def preprocess_observation(obs, prev_obs=None):
    """
    obs - a 210 x 160 x 3 ndarray representing an atari frame
    prev_obs - previous observation frame
    returns:
      a 3 x 210 x 160 normalized pytorch tensor
    """
    if obs is not None:
        return torch.from_numpy(obs).permute(2, 0, 1)/255.0


def validate(model, epsilon, render=False, nepisodes=1):
    assert hasattr(model, "get_action")
    torch.manual_seed(590060)
    np.random.seed(590060)
    model.eval()

    render = render and can_render

    if render:
        nepisodes = 1
        fig, ax = plt.subplots(1, 1)

    total_reward = 0
    steps_alive = []
    for i in range(nepisodes):
        env = gym.make("ALE/MsPacman-v5")
        obs = env.reset(seed=590060+i)[0]
        if render:
            im = ax.imshow(obs)
        observation = preprocess_observation( # 1 x 1 x ic x iH x iW
            obs).unsqueeze(0).unsqueeze(0)
        prev_state = None
        step = 0
        # play until the agent dies or we exceed 50000 observations
        while env.ale.lives() > 0 and step < 50000:
            action, prev_state = model.get_action(observation, epsilon, prev_state=prev_state)
            prev_obs = obs
            env_output = env.step(action)
            total_reward += env_output[1]
            if render:
                img = env_output[0]
                im.set_data(img)
                fig.canvas.draw_idle()
                plt.pause(0.1)
            obs_new = env_output[0]
            observation = preprocess_observation(
                obs_new, prev_obs).unsqueeze(0).unsqueeze(0)
            prev_obs = obs_new
            step += 1
        steps_alive.append(step)

    logging.info("Steps taken over each of {:d} episodes: {}".format(
        nepisodes, ", ".join(str(step) for step in steps_alive)))
    logging.info("Total return after {:d} episodes: {:.3f}".format(nepisodes, total_reward))