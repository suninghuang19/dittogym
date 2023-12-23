import gym
import taichi as ti
import morphmaze

ti.init(arch=ti.gpu)
gym.make("RUN_Fine-v0")

print("SUCCESS!")
print("You have successfully installed morphmaze and its dependencies.")
