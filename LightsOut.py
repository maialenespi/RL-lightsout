import gymnasium as gym
import pygame
import numpy as np
import pickle


class LightsOutEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = gym.spaces.MultiBinary((size, size))

        self.action_space = gym.spaces.Discrete(size**2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._target_state = np.zeros((size, size))

    def _get_obs(self):
        state_str = ''.join(map(str, self._state.flatten()))
        return int(state_str, 2)

    def _get_info(self):
        return np.sum(self._state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._state = np.array([[0, 0, 1, 1, 0],
                                [1, 0, 0, 1, 1],
                                [0, 0, 0, 1, 1],
                                [0, 1, 1, 0, 1],
                                [1, 1, 0, 1, 0]])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        X_COORD = action // self.size
        Y_COORD = action % self.size
        past_info = self._get_info()
        self._state[X_COORD][Y_COORD] += 1
        self._state[X_COORD][Y_COORD] %= 2

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for i, j in directions:
            new_x, new_y = X_COORD + i, Y_COORD + j

            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                self._state[new_x][new_y] += 1
                self._state[new_x][new_y] %= 2

        curr_info = self._get_info()
        terminated = np.array_equal(self._state, self._target_state)
        rew = past_info - curr_info
        reward = rew*10

        if terminated:
            reward = 250


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, self._state

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )

        # GRID LINES
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # GRID CELLS
        for y in range(self.size):
            for x in range(self.size):
                color = (128, 128, 128) if self._state[y][x] == 0 else (255, 255, 255)
                pygame.draw.rect(
                    canvas,
                    color,
                    (
                        x * pix_square_size,
                        y * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                )
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    (
                        x * pix_square_size,
                        y * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                    width=1,
                )


        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
