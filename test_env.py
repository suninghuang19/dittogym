import gym
import taichi as ti
import dittogym

ti.init(arch=ti.gpu)
gym.make("run-fine-v0")

print("SUCCESS!")
print("You have successfully installed dittogym and its dependencies.")
