from gym.envs.registration import register

# v0
register(
    id='SHAPE_MATCH-v0',
    entry_point='env.SHAPE_MATCH:SHAPE_MATCH',
    max_episode_steps=30,
)

# v0
register(
    id='RUN-v0',
    entry_point='env.RUN:RUN',
    max_episode_steps=800,
)

# v0
register(
    id='GROW-v0',
    entry_point='env.GROW:GROW',
    max_episode_steps=600,
)

# v0
register(
    id='KICK-v0',
    entry_point='env.KICK:KICK',
    max_episode_steps=500,
)

# v0
register(
    id='DIG-v0',
    entry_point='env.DIG:DIG',
    max_episode_steps=1000,
)

# v0
register(
    id='OBSTACLE-v0',
    entry_point='env.OBSTACLE:OBSTACLE',
    max_episode_steps=800,
)

# v0
register(
    id='CATCH-v0',
    entry_point='env.CATCH:CATCH',
    max_episode_steps=1200,
)

# v0
register(
    id='SLOT-v0',
    entry_point='env.SLOT:SLOT',
    max_episode_steps=1500,
)
