from scipy.stats import pearsonr


def test_accuracy(sa,su,d_k,m):
    """
    Test the accuracy of a machine learning model on a given dataset.

    Parameters
    ----------
    sa : ndarray
        The source array containing input data.
    su : ndarray
        The target array containing the true labels.
    d_k : int
        The dimension of the keys in the model.
    m : Model
        The machine learning model to be tested.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the given dataset, expressed as a percentage.
    """

    sa = sa.ravel()
    su = su.ravel()


    sa_est = d_k.T @ m
    sa_est = sa_est.ravel()

    min_length = min(len(sa_est), len(sa), len(su))
    sa_est = sa_est[:min_length]
    sa = sa[:min_length]
    su = su[:min_length]

    corr_sa, p_value = pearsonr(sa_est, sa)
    corr_su, p_value = pearsonr(sa_est, su)


    #print("Pearson correlation coefficient with sa:", corr_sa)
    #print("Pearson correlation coefficient with su:", corr_su)

    if corr_sa > corr_su:
        #print('The right speech has been selected :-)')
        #print('\n')
        return 1

    else:
        #print('The wrong speech has been selected :-(')
        #print("\n")
        return 0