import math
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accuracy(object):
    """ base class for accuracy calculation
    """

    def __init__(self):
        pass

    def calc(self, output, target):
        pass

    def prec(self):
        pass

    def result_str(self):
        pass


class MultiLabelAccuracy(Accuracy):
    """ class for multi label accuracy calculation
    """

    def __init__(self):
        self.accuracy = AverageMeter()

    # @profile
    def calc(self, output, target, fast=False):
        """Computes the precision of multi label prediction"""
        with torch.no_grad():
            if target.size(1) > 1:
                num_labels = target.sum(dim=1)
                valid_indices = torch.nonzero(num_labels)

                maxk = num_labels.max().int().item()
                if fast:
                    maxk = min(maxk, 10)

                maxk = max(1, maxk)
                topk, pred_topk = output.topk(maxk, dim=1, largest=True)

                n = valid_indices.size(0)
                pred = torch.zeros_like(output).cuda()

                if fast:
                    pred = pred.scatter(1, pred_topk, 1)
                else:
                    for i in range(n):
                        sample_index = valid_indices[i].item()
                        k = num_labels[sample_index].int().item()
                        pred[sample_index, pred_topk[sample_index, :k]] = 1

                pred = pred * target
                correct = pred.sum(dim=1)
                accuracy = correct[valid_indices] * 100. / num_labels[valid_indices]
                accuracy = accuracy.sum(dim=0).item()

                if n > 0:
                    accuracy /= n
                    self.accuracy.update(accuracy, n)
            else:
                pos_positive_count = ((output * target) > 0).sum()
                pos_negative_count = ((output * (torch.ones_like(target).cuda() - target)) < 0).sum()
                good_count = pos_positive_count + pos_negative_count
                n = output.size(0)
                accuracy = good_count * 100. / n
                self.accuracy.update(accuracy, n)

    def prec(self):
        return self.accuracy.avg

    def val_sum(self):
        return self.accuracy.sum

    def total_count(self):
        return self.accuracy.count

    def result_str(self):
        return 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=self.accuracy)


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class APMeter(Meter):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self):
        super(APMeter, self).__init__()
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.FloatTensor(torch.FloatStorage())

    def add(self, output, target, weight=None):
        """Add a new observation
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                (eg: a row [0, 1, 0, 1] indicates that the example is
                associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        if hasattr(torch, "arange"):
            rg = torch.arange(1, self.scores.size(0) + 1).float()
        else:
            rg = torch.range(1, self.scores.size(0)).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
        return ap


class mAPMeter(Meter):
    """
    The mAPMeter measures the mean average precision over all classes.
    The mAPMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self):
        super(mAPMeter, self).__init__()
        self.apmeter = APMeter()

    def reset(self):
        self.apmeter.reset()

    def add(self, output, target, weight=None):
        self.apmeter.add(output, target, weight)

    def value(self):
        return self.apmeter.value().mean()