
## Environment setup

- install pytorch according to https://pytorch.org/
- Get a modified version of gym-pybullet-drones
    ```shell
      git clone https://github.com/mahaitongdae/gym-pybullet-drones.git
      git checkout dev_haitong
      pip install -e .
    ```
  
## The other environment

```shell
pip install noise

## pyglet
git clone https://github.com/pyglet/pyglet.git
git checkout pyglet-1.5-maintainance
python setup.py install --user
```

## offline training

```shell
cd train
python main_pyb.py
```

## Sim-to-Real Transfer

### Real-world experiment requirements and instructions

- [Crazyflie 2.x](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/)
  - I used [crazyswarm](https://github.com/USC-ACTLab/crazyswarm) to control the crazyflies. For detailed setups, please
refer to [crazyswarm documentation](https://crazyswarm.readthedocs.io/en/latest/).
  - For logging the experiment, we need to hack the hardware. You pro