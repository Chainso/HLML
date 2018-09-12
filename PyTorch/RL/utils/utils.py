from scipy import signal

def discount(rewards, decay):
    """
    Computes the discounted returns for the given rewards

    rewards : The given rewards to compute the discounted returns for
    decay : The decay to discount the rewards by
    """
    return signal.lfilter([1], [1, -decay], rewards[::-1], axis=0)[::-1]