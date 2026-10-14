"""Shared laptop-side code for the LanderPi. The robot only publishes sensors and takes commands;
everything else runs here (see README)."""

from .rosbridge import RosbridgeClient

__all__ = ["RosbridgeClient"]
