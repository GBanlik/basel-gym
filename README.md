# Basel-Gym

Basel environment in accordance to the OpenAI Gym format to be solved with RL.

The environment seeks optimizing x, the disclosed value as a percentage of VaR, in the following equation:

![grc](https://latex.codecogs.com/gif.latex?GRC_t%20%3D%5Cmax%20%7Bk%5Cfrac%7B1%7D%7B60%7D%7B%5Csum_%7Bi%3D0%7D%5E%7B59%7D%7DxVaR_%7B1%2C%200.01%2C%20t%7D%2C%20xVaR_%7B1%2C%200.01%2C%20t%7D%5Csqrt%7B10%7D%7D "GRC")


| Environment | Action Space | State Space |
| --- | --- |  --- |
| Simple | Discrete(3000) | MultiDiscrete([250, 12, 8) |
| Complete| Box(0, 3) |Tuple(Discrete(250), Discrete(12), Discrete(8), Box(0, 3)) |
