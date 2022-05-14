from collections import defaultdict
from collections import deque
import math
import torch


class SmoothedValue(object):
    def __init__(self, window_size=500):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t", meter_creator=SmoothedValue):
        self.meters = defaultdict(meter_creator)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # we use avg rather than global avg. The reason is that some value
            # might be NaN in amp. amp will ignore it, but this function will
            # not ignore it. Thus, we use the avg rather than global avg
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.avg)
            )
        return self.delimiter.join(loss_str)


class MeanSigmaMetricLogger(object):
    def __init__(self, delimiter="\t", meter_creator=SmoothedValue):
        from src.tools.logger import MetricLogger
        self.mean_meters = MetricLogger(delimiter=delimiter,
                                        meter_creator=SmoothedValue)
        self.sq_meters = MetricLogger(delimiter=delimiter,
                                      meter_creator=SmoothedValue)

    def update(self, **kwargs):
        self.mean_meters.update(**kwargs)
        self.sq_meters.update(**dict((k, v * v) for k, v in kwargs.items()))

    def get_info(self):
        key_to_sigma = {}
        for k, v in self.mean_meters.meters.items():
            mean = v.global_avg
            mean_square = self.sq_meters.meters[k].global_avg
            sigma = mean_square - mean * mean
            sigma = math.sqrt(sigma)
            key_to_sigma[k] = sigma

        result = []
        for name, mean_meter in self.mean_meters.meters.items():
            result.append({'name': name,
                'global_avg': mean_meter.global_avg,
                'median': mean_meter.median,
                'count': mean_meter.count,
                'sigma': key_to_sigma[name]})
        return result

    def __str__(self):
        result = self.get_info()

        loss_str = []
        for info in result:
            loss_str.append(
                    "{}: {:.4f} ({:.4f}+-{:.4f})".format(
                        info['name'],
                        info['median'],
                        info['global_avg'],
                        info['sigma'])
            )
        return self.mean_meters.delimiter.join(loss_str)

