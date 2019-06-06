from scipy import signal

def discount(rewards, decay):
    """
    Computes the discounted returns for the given rewards

    rewards : The given rewards to compute the discounted returns for
    decay : The decay to discount the rewards by
    """
    return signal.lfilter([1], [1, -decay], rewards[::-1], axis=0)[::-1]

def normalize(arr, epsilon=1e-8):
    """
    Normalizes the given array using its mean and standard deviation

    arr : The array to be normalized
    epsilon : The value of epsilon to add to the standard deviation to prevent
              divide by 0 errors
    """
    return (arr - arr.mean()) / (arr.std() + epsilon)
