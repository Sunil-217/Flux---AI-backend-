"""AGI-inspired autonomous multi-agent orchestration.

An advanced autonomous multi-agent GenAI layer built from ordinary LLMs: a
planner, specialised agents, permissioned tools, layered memory, explicit task
state, self-evaluation and bounded recovery. It is not artificial general
intelligence and makes no claim to human-level general capability.

Modules are imported lazily by the route so that importing the app does not pull
in the whole agent stack on a small instance.
"""
