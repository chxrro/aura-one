import carla
import cv2
import numpy as np
import torch
from dqn_model import DQNAgent  # Assuming you have a DQN model implemented in dqn_model.py
import logging

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set up the simulation
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Initialize the DQN agent
    agent = DQNAgent(state_size=6, action_size=3)  # Adjust state_size and action_size as needed

    while True:
        # Get the current state from the simulation
        image = get_camera_image(vehicle)
        state = preprocess_image(image)
        speed = vehicle.get_velocity().length()
        state = np.append(state, [speed])

        # Get the action from the agent
        action = agent.act(state)

        # Apply the action to the vehicle
        apply_action(vehicle, action)

        # Get the reward and next state
        next_state, reward, done = get_reward_and_next_state(vehicle)

        # Train the agent
        agent.step(state, action, reward, next_state, done)

        logging.info(f"Action: {action}, Reward: {reward}, Done: {done}")

        if done:
            break

    # Clean up
    vehicle.destroy()

def get_camera_image(vehicle):
    # Function to get the camera image from the vehicle
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image = None

    def process_image(data):
        nonlocal image
        image = np.array(data.raw_data).reshape((data.height, data.width, 4))[:, :, :3]

    camera.listen(lambda data: process_image(data))
    while image is None:
        world.tick()
    camera.stop()
    camera.destroy()
    return image

def preprocess_image(image):
    # Function to preprocess the image for the DQN model
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    state = np.array(resized).reshape(1, 84, 84)
    return state

def apply_action(vehicle, action):
    # Function to apply the action to the vehicle
    control = carla.VehicleControl()
    if action == 0:
        control.throttle = 0.5
        control.steer = 0.0
    elif action == 1:
        control.throttle = 0.5
        control.steer = -0.5
    elif action == 2:
        control.throttle = 0.5
        control.steer = 0.5
    vehicle.apply_control(control)

def get_reward_and_next_state(vehicle):
    # Function to get the reward and next state
    image = get_camera_image(vehicle)
    next_state = preprocess_image(image)
    speed = vehicle.get_velocity().length()
    next_state = np.append(next_state, [speed])
    reward = compute_reward(vehicle)
    done = check_done(vehicle)
    return next_state, reward, done

def compute_reward(vehicle):
    # Enhanced reward logic
    reward = 1.0  # Base reward for moving forward

    # Check for collisions
    collision_sensor = world.get_blueprint_library().find('sensor.other.collision')
    collision = world.spawn_actor(collision_sensor, carla.Transform(), attach_to=vehicle)
    collision_data = None

    def on_collision(data):
        nonlocal collision_data
        collision_data = data

    collision.listen(lambda data: on_collision(data))
    world.tick()
    collision.stop()
    collision.destroy()

    if collision_data:
        reward -= 10.0  # Penalty for collision

    # Check for proximity to pedestrians
    pedestrians = world.get_actors().filter('walker.pedestrian.*')
    for pedestrian in pedestrians:
        distance = vehicle.get_location().distance(pedestrian.get_location())
        if distance < 5.0:
            reward -= 5.0  # Penalty for being too close to a pedestrian

    # Reward for staying within lanes
    lane_invasion_sensor = world.get_blueprint_library().find('sensor.other.lane_invasion')
    lane_invasion = world.spawn_actor(lane_invasion_sensor, carla.Transform(), attach_to=vehicle)
    lane_invasion_data = None

    def on_lane_invasion(data):
        nonlocal lane_invasion_data
        lane_invasion_data = data

    lane_invasion.listen(lambda data: on_lane_invasion(data))
    world.tick()
    lane_invasion.stop()
    lane_invasion.destroy()

    if lane_invasion_data:
        reward -= 2.0  # Penalty for lane invasion

    return reward

def check_done(vehicle):
    # Termination condition
    done = False

    # Check for collisions
    collision_sensor = world.get_blueprint_library().find('sensor.other.collision')
    collision = world.spawn_actor(collision_sensor, carla.Transform(), attach_to=vehicle)
    collision_data = None

    def on_collision(data):
        nonlocal collision_data
        collision_data = data

    collision.listen(lambda data: on_collision(data))
    world.tick()
    collision.stop()
    collision.destroy()

    if collision_data:
        done = True  # Terminate if collision occurs

    return done

if __name__ == "__main__":
    main()