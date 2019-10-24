class ConsecutiveNonImprovment():
    def __init__(self, num_epochs):
        """
        param: num_epochs

        Number of epochs which are considered, if there was no
        improvement in the last epochs we stop.
        """
        self.num_epochs = num_epochs

    def __call__(self, losses):
        """
        Returns true if stopping criterion is reached.
        """
        if len(losses) < self.num_epochs:
            return False
        improvements = 0
        consider = losses[-(self.num_epochs + 1):]
        for i, item in enumerate(consider[1:]):
            if consider[i] > item:
                improvements += 1
        return improvements == 0
