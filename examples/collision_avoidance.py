"""Collision avoidance example: multi-agent safety."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi_agent.collision_avoidance import Position, Velocity, CollisionAvoidance


def main():
    avoidance = CollisionAvoidance(safety_radius=2.0)
    pos1 = Position(x=0.0, y=0.0)
    pos2 = Position(x=1.0, y=1.0)
    vel1 = Velocity(vx=1.0, vy=0.0)
    vel2 = Velocity(vx=-1.0, vy=0.0)
    result = avoidance.check_collision(pos1, pos2, vel1, vel2)
    print(f"Collision Risk: {result}")
    print(f"Safety Radius: {avoidance.safety_radius}m")


if __name__ == "__main__":
    main()
