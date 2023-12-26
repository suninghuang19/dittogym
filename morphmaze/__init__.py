from gym.envs.registration import register
from .shape_match import SHAPE_MATCH
from .run import RUN
from .grow import GROW
from .kick import KICK
from .dig import DIG
from .obstacle import OBSTACLE
from .catch import CATCH
from .slot import SLOT
import os
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


register(
    id='SHAPE_MATCH_Coarse-v0',
    entry_point='morphmaze:SHAPE_MATCH',
    kwargs={'action_dim': 2*32**2, 'action_res_resize': 2, 'cfg_path':current_directory + '/cfg/shape_match.json'},
    max_episode_steps=30,
)

register(
    id='SHAPE_MATCH_Fine-v0',
    entry_point='morphmaze:SHAPE_MATCH',
    kwargs={'action_dim': 2*64**2, 'action_res_resize': 1, 'cfg_path':current_directory + '/cfg/shape_match.json'},
    max_episode_steps=30,
)



register(
    id='RUN_Coarse-v0',
    entry_point='morphmaze:RUN',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/run.json'},
    max_episode_steps=800,
)

register(
    id='RUN_Fine-v0',
    entry_point='morphmaze:RUN',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/run.json'},
    max_episode_steps=800,
)



register(
    id='GROW_Coarse-v0',
    entry_point='morphmaze:GROW',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/grow.json'},
    max_episode_steps=600,
)

register(
    id='GROW_Fine-v0',
    entry_point='morphmaze:GROW',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/grow.json'},
    max_episode_steps=600,
)



register(
    id='KICK_Coarse-v0',
    entry_point='morphmaze:KICK',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/kick.json'},
    max_episode_steps=500,
)

register(
    id='KICK_Fine-v0',
    entry_point='morphmaze:KICK',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/kick.json'},
    max_episode_steps=500,
)



register(
    id='DIG_Coarse-v0',
    entry_point='morphmaze:DIG',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/dig.json'},
    max_episode_steps=1000,
)

register(
    id='DIG_Fine-v0',
    entry_point='morphmaze:DIG',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 2, 'cfg_path':current_directory + '/cfg/dig.json'},
    max_episode_steps=1000,
)



register(
    id='OBSTACLE_Coarse-v0',
    entry_point='morphmaze:OBSTACLE',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/obstacle.json'},
    max_episode_steps=800,
)

register(
    id='OBSTACLE_Fine-v0',
    entry_point='morphmaze:OBSTACLE',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/obstacle.json'},
    max_episode_steps=800,
)



register(
    id='CATCH_Coarse-v0',
    entry_point='morphmaze:CATCH',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/catch.json'},
    max_episode_steps=1200,
)

register(
    id='CATCH_Fine-v0',
    entry_point='morphmaze:CATCH',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/catch.json'},
    max_episode_steps=1200,
)



register(
    id='SLOT_Coarse-v0',
    entry_point='morphmaze:SLOT',
    kwargs={'action_dim': 2*8**2, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/slot.json'},
    max_episode_steps=1500,
)

register(
    id='SLOT_Fine-v0',
    entry_point='morphmaze:SLOT',
    kwargs={'action_dim': 2*16**2, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/slot.json'},
    max_episode_steps=1500,
)