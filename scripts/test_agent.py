"""
Example agent script. Interact with the agent in semantic I/O.

Usage:
    $ python3 test.py
    input: [{"intent":"greeting"}]
    [{"intent": "request", "slot": "moviename"}]

"""
from miuds import Agent

agent = Agent(dataset='movie.json', database='movie.db')
agent.semantic_mode()
while True:
    print(agent(input('input: ')))
