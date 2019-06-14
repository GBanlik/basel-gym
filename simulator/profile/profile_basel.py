import numpy as np

from math import sqrt

from typing import Deque, Type

#from scipy.special import ndtr
from scipy.stats.distributions import norm

from simulator.profile.profile_base import SimulationProfileBase
from simulator.distribution.distribution_base import SimulationDistributionBase
from simulator.monitor.monitor_base import SimulationMonitorBase
from simulator.monitor.monitor_basel import BaselSimulationMonitor
from utils.utils_decorators import inputDecorators


SQRT_10 = sqrt(10)

class BaselSimulationProfile(SimulationProfileBase):
    def __init__(self, dist: Type[SimulationDistributionBase], monitor: Type[SimulationMonitorBase], config: dict = {}):
        super().__init__(dist, monitor, config)

        self._confidence_level: float = 0.99
        self._normal_mean: float = 0
        self._normal_stddev = 1

        config_dist = config.get('returns_distribution', None)

        if not config_dist is None:
            self._confidence_level = config_dist.get("confidence_level", self._confidence_level)
            self._normal_mean = config_dist.get("mean", self._normal_mean)
            self._normal_stddev = config_dist.get("std", self._normal_stddev)       

        self._normal_var: float = config.get("normal_var", norm.ppf(self._confidence_level, self._normal_mean, self._normal_stddev))
        self._normal_var10: float = config.get("normal_var10", -self._normal_var * sqrt(10))
        # the maximum allowed reported/disclosued value
        self._max_report_value = config.get("max_report_value", 3)
        print(self._confidence_level, self._normal_mean, self._normal_stddev, self._normal_var)
        self._k_multipliers: np.ndarray = np.array([
            [3, 3, 3, 3, 3, 3.4, 3.5, 3.65, 3.75, 3.85, 4, 10000],
            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]])

    def performTransition(self, daily_return: np.ndarray, sim_state: np.ndarray) -> None:
        monitor: BaselSimulationMonitor = self._monitor
        sim_num = sim_state[0]
        day = sim_state[1]
        done = sim_state[2]
        basel_record_categories = BaselSimulationMonitor.BaselRecordCategory

        # pointers to cleanup the code
        rc_kmul_idx = basel_record_categories.KMULTIPLIERS_INDECES
        rc_kmul_value = basel_record_categories.KMULTIPLIERS_VALUE
        rc_ec = basel_record_categories.EXCEEDENCES
        rc_bk = basel_record_categories.BANKRUPTCY

        # helper to clean the code
        fetch_record = lambda key, id = None: monitor.record(key) if id is None else  monitor.record(key)[id]

        current_k_idx: np.array = fetch_record(rc_kmul_idx, sim_num)
        current_k: np.array = fetch_record(rc_kmul_value, sim_num)
        current_ecs: np.array = fetch_record(rc_ec, sim_num)
        # fetch the previous' period's bankruptcy state, as its a permanent state
        bankruptcy: np.array = monitor.record(rc_bk)[sim_num]
                
        disclosure_history: np.array = monitor.disclosure_history

        # turn observations into rows to be fed onto getAction (Discrete Simulation)
        # Subtract day as the estimated distribution 250 equals 0
        env_obs: np.ndarray = np.vstack((current_k_idx.T, current_ecs.T, np.full(current_ecs.T.shape, 249 - day))).astype(np.int32)
   
        #disclosure = normvar * disclosed value
        disclosure: np.array = self.distribution.getAction(env_obs)

        #bankrupt states should always report the maximum value so as to avoid bankruptcy
        disclosure[current_ecs == 10] = self._max_report_value

        reported_value: np.array =  disclosure * self._normal_var
        #increment day to reject is null record
        reported_mean = np.ones(shape=(disclosure_history[0].shape), dtype=float) if \
            day == 249 else disclosure_history[(day+1):].mean(axis=0)
        #print(reported_value, disclosure, self._normal_var)
        #BC = MRC is below the loss, MRC = mean(last_60_disclosure) * kMul * sqrt(10)
        bankruptcy += (reported_mean.T * current_k * SQRT_10 < daily_return)
        bankruptcy = np.minimum(bankruptcy, 1)

        # EC = return < disclosure , -100 for bankruptcy states
        if day == 249:
            current_ecs = (daily_return < -reported_value) * 1
        else:            
            current_ecs += (daily_return < -reported_value) * 1

        current_ecs += (bankruptcy) * 11
        current_ecs = np.minimum(current_ecs.astype(int), 11)

        monitor.record(rc_ec)[sim_num] = current_ecs
        monitor.record(rc_bk)[sim_num] = bankruptcy

        #TODO Add the portfolio's invested amount / return
        # the invested amount corresponds to the portfolio + mrc in proportion
        monitor.record(basel_record_categories.RETURN_DAILY)[day] = daily_return
        monitor.addRecord(category_key=basel_record_categories.DISCLOSURE, record=disclosure, record_key=day)

        if(done):
            #review the k Multiplier applicable on the following year
            reviewed_k_idx = (self._k_multipliers[1, current_ecs]).astype(int)
            reviewed_k_val: np.array = self._k_multipliers[0, current_ecs]

            monitor.record(rc_kmul_idx)[sim_num+1] = reviewed_k_idx
            monitor.record(rc_kmul_value)[sim_num+1] = reviewed_k_val
            monitor.record(rc_bk)[sim_num+1] = bankruptcy

            # store the year's average disclosure
            monitor.record(basel_record_categories.DISCLOSURE_YEAR_MEAN)[sim_num] = reported_mean

            # store the year's average return
            avg_return: np.array = monitor.record(basel_record_categories.RETURN_DAILY).mean(axis=0)
            monitor.record(basel_record_categories.RETURN_YEAR)[sim_num] = avg_return
            

        