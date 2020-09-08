def add_boruta_feature(boruta, n):
    '''
    Manually adds the feature at index n to a boruta model
    :param boruta: an SKLearn boruta model
    :param feature: an int for the desired feature index
    :return: the sklearn model with the feature at index n changed to true in boruta.support_
    '''

    boruta.support_[n] = True
    return boruta


def remove_boruta_feature(boruta, n):
    '''
    Manually removes the feature at index n to a boruta model
    :param boruta: an SKLearn boruta model
    :param feature: an int for the desired feature index
    :return: the skearn model with the feature at index n changed to true
    '''
    boruta.support_[n] = False
    boruta.support_weak_ = False
    return boruta


def main():
    pass


if __name__ == '__main__':
    import sys

    sys.exit(main())
