import numpy as np
class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def sample_prob_vec(pi):
    bucket = np.random.choice(len(pi), p=pi)
    x = np.random.random()

    left_bucket = max(0, bucket - 1)
    right_bucket = min(len(pi) - 1, bucket + 1)

    val = (pi[left_bucket]*left_bucket*x + pi[bucket]*bucket + pi[right_bucket]*(1-x)*right_bucket)
    val /= (pi[left_bucket] + pi[bucket] + pi[right_bucket])

    val /= len(pi) - 1
    return val

def sample_from_bucket(pi, bucket):
    x = np.random.random()

    left_bucket = max(0, bucket - 1)
    right_bucket = min(len(pi) - 1, bucket + 1)

    val = (pi[left_bucket]*left_bucket*x + pi[bucket]*bucket + pi[right_bucket]*(1-x)*right_bucket)
    val /= (pi[left_bucket] + pi[bucket] + pi[right_bucket])

    val /= len(pi) - 1
    return val

    #return int(np.ceil(100*val))