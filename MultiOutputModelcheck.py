#Class to check if input parameters are suitable for MultiOutputProblem
class MultiCheck(object):
    def __init__(self, model, times, values):

        self.model = model

        self._values = pints.matrix2d(values)
        self._times = pints.vector(times)

        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array not right shape')
        
MultiCheck(model, time, values)
