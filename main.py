from agent import Agent

PARAM_SET = "flappybird"
TRAIN = False
RENDER = True

agent = Agent(PARAM_SET)
agent.run(is_training = TRAIN, render = RENDER)
