# Morphological Maze

Official Gym Environment Implementation of ***[Morphological Maze: Control Reconfigurable Soft Robots with Fine-grained Morphology Change](https://morphologicalmaze.github.io/)***

*****

To create environment, simply run the following command in the root:
```python
pip install .
```

You can run the following command to check if successfully installed:
```python
python test.py
```



### Benchmark Introduction

**<font color=red>Videos can be found at [here](https://morphologicalmaze.github.io/).</font>**

- **SHAPE MATCH:** The robot is initialized as a circle in a zero-gravity environment. It has to alter its shape to mirror predefined geometric or alphabetical form. The score is calculated on the basis of the congruence between the robot's current shape and the target shape.
- **RUN:** The robot is initialized on a plain and the task requires a robot to move forward to the greatest possible distance within a stipulated period. The score is governed by the distance traversed and the speed maintained.
- **KICK:** The robot is initialized as a circle and is designed to kick a square target to the maximum possible distance. Ground friction prevents simple pushing, therefore the robot is required to use a flipping and rolling technique.
- **DIG:** The robot is initially in the shape of a circle placed on the top of a soil-filled container, aiming to reach a target located at the bottom-right corner.
- **GROW:** The robot is initialized as a square on the ground and is required to extend its superior segment to reach a target represented by a purple dot. Multiple obstacles hinder the direct route to the target. The reward function measures the distance between the target point and the robot.
- **OBSTACLE:** The robot, in the shape of a square, faces an obstacle in its path while the task is to move forward. The score is based solely on how far the obstacle is bypassed.
- **CATCH:** The circular robot, placed outside a cube, is tasked with manipulating a small square target within the cube to a specific point. The score is computed considering the distance between the robot and the cube, as well as the cube and the final point.
- **SLOT:** The circular robot, initially outside a box with only a narrow slot to get inside, must squeeze its body through the slot and manipulate a cap target on top of the box. The reward function measures the distance between the robot and the cap, and whether the cap was successfully removed from the box.

![](./teaser/teaser.png)

**MorphMaze ENV Parameters:**

|    Task     | Observation Resolution | Coarse Action Resolution (ENV-Coarse) | Fine Action Resolution (ENV-Fine) | Interpolated Action to X Resolution Field |
| :---------: | :--------------------: | :-----------------------------------: | :-------------------------------: | :---------------------------------------: |
| SHAPE_MATCH |      (64, 64, 3)       |              (32, 32, 2)              |            (64, 64, 2)            |                (64, 64, 2)                |
|     RUN     |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (64, 64, 2)                |
|    KICK     |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (64, 64, 2)                |
|     DIG     |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (32, 32, 2)                |
|    GROW     |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (64, 64, 2)                |
|  OBSTACLE   |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (64, 64, 2)                |
|    CATCH    |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (64, 64, 2)                |
|    SLOT     |      (64, 64, 3)       |               (8, 8, 2)               |            (16, 16, 2)            |                (64, 64, 2)                |

