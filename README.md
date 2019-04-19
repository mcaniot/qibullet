# qibullet [![Build Status](https://api.travis-ci.org/ProtolabSBRE/qibullet.svg?branch=master)](https://travis-ci.org/ProtolabSBRE/qibullet) [![pypi](https://img.shields.io/pypi/v/qibullet.svg)](https://pypi.org/project/qibullet/)

__Bullet-based__ python simulation for __SoftBank Robotics'__ robots.

<p align="middle">
	<img src="ressources/short_top_cam.gif" width="46%" />
	<img src="ressources/pepper_depth_camera.gif" width="49%" />
</p>
<p align="middle">
	<img src="ressources/pepper_moveTo.gif" width="33%" />
	<img src="ressources/ros_compat.gif" width="62%" />
</p>

## Installation

The following modules are required:
* __numpy__
* __pybullet__

The __qibullet__ module can be installed via pip, for python 2.7 and python 3:
```bash
pip install --user qibullet
```

## Usage
Please note that only the Pepper robot is currently handled by this module. A robot can be spawn via the SimulationManager class:
```python
from qibullet import SimulationManager

if __name__ == "__main__":
    simulation_manager = SimulationManager()

    # Launch a simulation instances, with using a graphical interface.
    # Please note that only one graphical interface can be launched at a time
    client_id = simulation_manager.launchSimulation(gui=True)

    # Spawning a virtual Pepper robot, at the origin of the WORLD frame, and a
    # ground plane
    pepper = simulation_manager.spawnPepper(
        client_id,
        translation=[0, 0, 0],
        quaternion=[0, 0, 0, 1],
        spawn_ground_plane=True)
```

Or using loadRobot from the PepperVirtual class if you already have a simulated environment:
```python
    pepper = PepperVirtual()

    pepper.loadRobot(
      translation=[0, 0, 0],
      quaternion=[0, 0, 0, 1],
      physicsClientId=client_id)
```

More snippets can be found in the [examples folder](examples/).

## Documentation
The qibullet __API documentation__ can be found [here](https://protolabsbre.github.io/qibullet/api/). The documentation can be generated via the following command (the __doxygen__ package has to be installed beforehand, and the docs folder has to exist):
```bash
cd docs
doxygen
```

## Troubleshooting

### OpenGL driver
If you encounter the message:
> Workaround for some crash in the Intel OpenGL driver on Linux/Ubuntu

Your computer is using the Intel OpenGL driver. Go to __Software & Updates__, __Additional Drivers__, and select a driver corresponding to your GPU.
