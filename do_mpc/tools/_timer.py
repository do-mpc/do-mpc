import time
import numpy as np
import matplotlib.pyplot as plt


class Timer:

    def __init__(self, name='timer', unit='ms'):
        self.name = name

        self.unit = unit
        if unit =='ms':
            self.factor = 1000
        elif unit == 's':
            self.factor = 1
        elif unit == 'min':
            self.factor = 1/60
        elif unit == 'h':
            self.factor = 1/3600
        else:
            raise Exception('unit must be in [ms, s, min, h].')

        self.t_start = None
        self.t_list = []

    def tic(self):
        self.t_start = time.time()

    def toc(self):
        assert self.t_start, 'Cannot call toc before toc.'

        t_now = time.time()
        self.t_list.append(t_now-self.t_start)

        self.t_start = None

    def info(self):
        t_arr = np.round(np.array(self.t_list)*self.factor)
        t_mean = np.mean(t_arr)
        t_min = np.min(t_arr)
        t_max = np.max(t_arr)
        t_var = np.std(t_arr)

        msg = 'Average runtime {avg}+-{var}{unit}. Fastest run {min}{unit}, slowest run {max}{unit}.'
        print(msg.format(avg=t_mean, var=t_var, unit=self.unit, min=t_min, max=t_max))

    def hist(self, *args, **kwargs):
        t_arr = np.round(np.array(self.t_list)*self.factor)
        t_mean = np.mean(t_arr)

        fig, ax = plt.subplots()
        ax.axvline(t_mean, color='black')
        ax.hist(t_arr, *args, **kwargs)
        ax.set_xlabel('time in {}'.format(self.unit))
        ax.set_ylabel('number of instances')

        plt.show()

        return fig, ax

        
