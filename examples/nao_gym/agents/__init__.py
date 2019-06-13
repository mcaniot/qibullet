# noqa: F401
# flake8: noqa
# ---- register agents ----------
from agents import agent_register
from agents.agents_kerasrl import KerasCEMAgent, KerasDDPGAgent,\
                           KerasDDQNAgent, KerasDQNAgent, KerasNAFAgent

agent_register.register(
    id='BaselinesDQNAgent-v0',
    entry_point='agents_baselines:BaselinesDQNAgent'
)

agent_register.register(
    id='KerasCEMAgent-v0',
    entry_point='agents.agents_kerasrl:KerasCEMAgent'
)

agent_register.register(
    id='KerasDDPGAgent-v0',
    entry_point='agents.agents_kerasrl:KerasDDPGAgent'
)

agent_register.register(
    id='KerasDDQNAgent-v0',
    entry_point='agents.agents_kerasrl:KerasDDQNAgent'
)

agent_register.register(
    id='KerasDQNAgent-v0',
    entry_point='agents.agents_kerasrl:KerasDQNAgent'
)

agent_register.register(
    id='KerasNAFAgent-v0',
    entry_point='agents.agents_kerasrl:KerasNAFAgent'
)
