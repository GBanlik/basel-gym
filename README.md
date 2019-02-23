# Basel-Gym

Basel environment in accordance to the OpenAI Gym format to be solved with RL.

The environment seeks optimizing the following equation:

![grc](https://latex.codecogs.com/gif.latex?GRC_t%20%3Dmax%28k%5Cfrac%7B1%7D%7B60%7D%5Csum%5E%7B59%7D_%7Bi%3D0%7DVaR_%7B1%2C0.01%2Ct-i%7D%5Csqrt%7B10%7D%2CVaR_%7B1%2C0.01%2C%20t%7D%5Csqrt%7B10%7D%29 "GRC")


| Environment | Action Space | State Space |
| --- | --- |  --- |
| Simple | Discrete(3000) | MultiDiscrete([250, 12, 8) |
| Complete| Box(0, 3) |Tuple(Discrete(250), Discrete(12), Discrete(8), Box(0, 3)) |