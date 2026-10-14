"""Robot actions a language model can call (the task console's chat, projects/navigation/assistant.py).
One registry for all projects: each project keeps its tools in its own module and registers them here.

    from landerpi.tools import ToolError, tool

    @tool("drive_to_object", "Drive to an object of the room and stop in front of it, facing it.",
          {"object_id": {"type": "integer", "description": "the object's [id] in the room list"}},
          moves_robot=True)
    def drive_to_object(ctx, object_id):
        ...                                    # ctx: Context (below)
        return {"result": "at cabinet, 0.21 m in front of it"}

    schemas()                 # the tools in OpenAI's "tools" format, for the model
    run(name, args, ctx)      # calls one, returns a JSON-able dict for the model (never raises)

Rules for every tool (the baseline, agreed with the user 2026-10-07):
  - name: verb_object in snake_case. The description is written for the model: what the tool does, when to
    use it, what it needs first (e.g. "the robot must be localized").
  - moves_robot=True (base or arm): refused unless the console is Armed (ctx.armed); the model then asks the
    user to arm. STOP / Space / disarming in the console stop a running motion (ctx.stopped() turns True).
  - Long actions (driving, grasping) wait until they are done and report the outcome, so the model can
    chain steps ("to the door, then to the cabinet").
  - The result is a small dict with a "result" sentence and numbers where useful; failures raise ToolError
    with a sentence the model can pass on to the user ("the robot is not localized: press ⟳ Localize").
  - Objects are referred to by the [id] of the room's object list that the model gets with every message.
    Ids change when a room is relabelled (scene_understanding): never store them between conversations.
  - A project's tools live in projects/<name>/..._tools.py and are listed in assistant.TOOL_MODULES.
"""

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """A failure the model should explain to the user (the message is passed on as it is)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema "properties"
    required: list
    moves_robot: bool
    run: Callable


@dataclass
class Context:
    """What a tool gets: whether the console is Armed, a check for STOP, and the services of the process the
    chat runs in (navigation: "navigator"; other projects add theirs)."""
    armed: bool = False
    stopped: Callable[[], bool] = lambda: False
    services: dict = field(default_factory=dict)


REGISTRY: dict = {}


def tool(name, description, parameters=None, required=None, moves_robot=False):
    """Decorator: registers fn(ctx, **args) as a tool. required: default = all parameters."""
    parameters = parameters or {}

    def register(fn):
        if name in REGISTRY and REGISTRY[name].run is not fn:
            raise ValueError(f"tool {name} is registered twice")
        REGISTRY[name] = Tool(name, description, parameters,
                              list(parameters) if required is None else required, moves_robot, fn)
        return fn
    return register


def schemas():
    return [{"type": "function", "function": {
        "name": t.name,
        "description": t.description + (" Moves the robot: the console must be Armed." if t.moves_robot else ""),
        "parameters": {"type": "object", "properties": t.parameters, "required": t.required}}}
        for t in REGISTRY.values()]


def run(name, args, ctx: Context) -> dict[str, Any]:
    t = REGISTRY.get(name)
    if t is None:
        return {"error": f"there is no tool {name}"}
    if t.moves_robot and not ctx.armed:
        return {"error": "the robot is not armed: ask the user to switch on Armed (top right) and try again"}
    try:
        return t.run(ctx, **(args or {}))
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:  # wrong or missing arguments from the model
        return {"error": f"wrong arguments for {name}: {e}"}
    except Exception as e:  # a bug: tell the model, keep the chat alive, log it
        traceback.print_exc()
        return {"error": f"{name} failed ({type(e).__name__}: {e})"}
