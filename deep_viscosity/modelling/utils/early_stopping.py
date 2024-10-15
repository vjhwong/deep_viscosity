class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        EarlyStopping constructor.

        Parameters:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        """
        Call method to check if early stopping criteria are met.

        Parameters:
            metric (float): The monitored metric (e.g., validation loss).
        """
        if self.best_score is None:
            self.best_score = metric
        elif metric < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Validation metric did not improve. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f'Validation metric improved to {metric:.6f}. Resetting counter.')