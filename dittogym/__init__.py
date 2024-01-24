from gym.envs.registration import register
from .shapematch import shapematch
from .run import run
from .grow import grow
from .kick import kick
from .dig import dig
from .obstacle import obstacle
from .catch import catch
from .slot import slot
import os
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


register(
    id='shapematch-coarse-v0',
    entry_point='dittogym:shapematch',
    kwargs={'action_res': 32, 'action_res_resize': 2, 'cfg_path':current_directory + '/cfg/shapematch-coarse.json'},
    max_episode_steps=30,
)

register(
    id='shapematch-fine-v0',
    entry_point='dittogym:shapematch',
    kwargs={'action_res': 64, 'action_res_resize': 1, 'cfg_path':current_directory + '/cfg/shapematch-fine.json'},
    max_episode_steps=30,
)



register(
    id='run-coarse-v0',
    entry_point='dittogym:run',
    kwargs={'action_res': 8, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/run-coarse.json'},
    max_episode_steps=800,
)

register(
    id='run-fine-v0',
    entry_point='dittogym:run',
    kwargs={'action_res': 16, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/run-fine.json'},
    max_episode_steps=800,
)



register(
    id='gorw-coarse-v0',
    entry_point='dittogym:grow',
    kwargs={'action_res': 8, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/grow-coarse.json'},
    max_episode_steps=600,
)

register(
    id='grow-fine-v0',
    entry_point='dittogym:grow',
    kwargs={'action_res': 16, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/grow-fine.json'},
    max_episode_steps=600,
)



register(
    id='kick-coarse-v0',
    entry_point='dittogym:kick',
    kwargs={'action_res': 8, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/kick-coarse.json'},
    max_episode_steps=500,
)

register(
    id='kick-fine-v0',
    entry_point='dittogym:kick',
    kwargs={'action_res': 16, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/kick-fine.json'},
    max_episode_steps=500,
)



register(
    id='dig-coarse-v0',
    entry_point='dittogym:dig',
    kwargs={'action_res': 8, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/dig-coarse.json'},
    max_episode_steps=1000,
)

register(
    id='dig-fine-v0',
    entry_point='dittogym:dig',
    kwargs={'action_res': 16, 'action_res_resize': 2, 'cfg_path':current_directory + '/cfg/dig-fine.json'},
    max_episode_steps=1000,
)



register(
    id='obstacle-coarse-v0',
    entry_point='dittogym:obstacle',
    kwargs={'action_res': 8, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/obstacle-coarse.json'},
    max_episode_steps=800,
)

register(
    id='obstacle-fine-v0',
    entry_point='dittogym:obstacle',
    kwargs={'action_res': 16, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/obstacle-fine.json'},
    max_episode_steps=800,
)



register(
    id='catch-coarse-v0',
    entry_point='dittogym:catch',
    kwargs={'action_res': 8, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/catch-coarse.json'},
    max_episode_steps=1200,
)

register(
    id='catch-fine-v0',
    entry_point='dittogym:catch',
    kwargs={'action_res': 16, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/catch-fine.json'},
    max_episode_steps=1200,
)



register(
    id='slot-coarse-v0',
    entry_point='dittogym:slot',
    kwargs={'action_res': 8, 'action_res_resize': 8, 'cfg_path':current_directory + '/cfg/slot-coarse.json'},
    max_episode_steps=1500,
)

register(
    id='slot-fine-v0',
    entry_point='dittogym:slot',
    kwargs={'action_res': 16, 'action_res_resize': 4, 'cfg_path':current_directory + '/cfg/slot-fine.json'},
    max_episode_steps=1500,
)