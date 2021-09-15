import numpy as np
import random
from enum import Enum
from scipy import stats


class DistributionType(Enum):
    normal = 1
    exponential = 2
    uniform = 3
    scheduled = 4
    immediate = 5
    kernel_density_estimate = 6
    deterministic = 7
    empirical = 8
    laplace = 9


class Distribution:
    """
    a class to capture supported distributions that can be fit to data
    """

    def __init__(self, dist_type=DistributionType.exponential, **kwargs):
        fit = kwargs.get('fit', False)
        if not fit:
            if dist_type == DistributionType.normal:
                self._param = [kwargs.get('mean', 5), kwargs.get('sd', 1)]
            elif dist_type == DistributionType.exponential:
                self._param = [kwargs.get('rate', 1)]
            elif dist_type == DistributionType.uniform:
                self._param = [kwargs.get('low', 0), kwargs.get('high', 1)]
            elif dist_type == DistributionType.scheduled:
                self._param = [kwargs.get('mintime', 0)]
            elif dist_type == DistributionType.deterministic:
                self._param = [kwargs.get('time', 0)]
            elif dist_type == DistributionType.laplace:
                self._param = [kwargs.get('location', 0), kwargs.get('scale', 1)]
            else:  # immediate transition
                self._param = [0]
                dist_type = DistributionType.immediate
        else:  # fit distributions to data
            # Assuming that the values are stored in a 1-d vector called "values"
            self._param = kwargs.get('values', [0, 0, 1])
            if dist_type == DistributionType.empirical:
                pass
            else:
                self._dist = stats.gaussian_kde(self._param)
                dist_type = DistributionType.kernel_density_estimate

        self._name = dist_type
        self.type = dist_type
        # print "creating {} distribution with {} values".format(dist_type, self._param)



    def __repr__(self):
        return "{} distribution (params:{})".format(self._name, self._param)

    def sample(self, **kwargs):
        """
        Samples a random value from the distribution that is used.
        :return: a sample from the distribution
        """
        if self._name == DistributionType.normal:
            return random.gauss(self._param[0], self._param[1])
        elif self._name == DistributionType.exponential:
            return random.expovariate(self._param[0])
        elif self._name == DistributionType.immediate:
            return 0
        elif self._name == DistributionType.scheduled:
            current_time = kwargs.get('time', 0)
            return max(self._param[0], current_time) - current_time
        elif self._name == DistributionType.kernel_density_estimate:
            return abs(self._dist.resample(size=1)[0][0])
        elif self._name == DistributionType.deterministic:
            return self._param[0]
        elif self._name == DistributionType.empirical:
            # draw one of the past samples uniformly
            return self._param[random.randint(0, len(self._param) - 1)]

        else:  # assume uniform distribution
            return random.uniform(self._param[0], self._param[1])

    def get_mean(self):
        if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
            return np.mean(self._param)
        elif self._name == DistributionType.exponential:
            return 1 / self._param[0]
        elif self._name == DistributionType.deterministic:
            return self._param[0]
        elif self._name == DistributionType.immediate:
            return 0
        elif self._name == DistributionType.laplace:
            return self._param[0]
        else:
            print("please implement for the Mean {} - debug me!".format(self._name))
            return -1

    def get_CV(self):
        if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
            if np.mean(self._param) == 0:
                print("zero mean for the CV calculation - debug me!")
            else:
                return np.std(self._param) / np.mean(self._param)
        elif self._name == DistributionType.exponential:
            return 1
        elif self._name == DistributionType.deterministic:
            return 0
        elif self._name == DistributionType.immediate:
            return 0
        else:
            print("please implement for the CV {} - debug me!".format(self._name))
            return -1


class Token:
    def __init__(self, id_, type_, history, enqueue_time):

        self.id = id_
        self.type = type_ #includes sequence of types and times - only changes
        self.history = history #includes sequence and times - only changes


    def update_data(self, type_, history):
        self.type = type_
        self.history = history

class Resource:
    def __init__(self, name, capacity, processing_function, policy):
    #init is always an empty queue.
        self.name = name #unique name for resource type
        self.capacity = capacity #resource capacity, het. resources - todo: can be dynamically changed (?)
        self.queue = [] #list of waiting tokens
        self.service = [] # tokens in service
        #self.completed  = [] #buffer of finished tokens
        #should we define basic rule for token selection as attribute or method (same for processing?)
        self.processing_function = processing_function
        self.policy = policy
    def add_to_queue(self, token):
        self.queue.append(token)
        return
    def remove_from_queue(self, token):
        self.queue.remove(token)
        return

    #select is a method of the queue that can be learned or given (ranking function should be given)
    def select_token(self):
        #can use enhanced ranking from data
        ordering = []
        for token in self.queue:
            ordering.append((token, self.policy(token)))
        #order the tokens according to their rank score and return the highest in the list
        #tie break randomly.

        return #token of choice

    def assign_processing_time(self, token): #=default distribution/empirical):
        #draw duration from distribution - unique to Resource-Token combination + processing function that we learn.
        #can use enhanced processing function (see Andreas code)

        return self.processing_function(token)





