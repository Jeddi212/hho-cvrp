from model.export import Export
from controller.optimizer import run
from model.parameter import Parameter


def conf(num, pop, iterate, names, result_directory=None, optimizer=None):
    """
    Parameter configuration

    Parameters
    ----------
    num : int
        number of run
    pop : int
        population size
    iterate : int
        number of iteration
    names : list of str
        list of instance name to be tested
    result_directory : str
        path of root directory for result files
    optimizer : list of str
        list of optimizer name to be tested

    Returns
    -------
    N/A
    """

    # Collection of params
    params = Parameter(num, pop, iterate, names, result_directory, optimizer)

    # Choose whether to Export the results in different formats
    export_flags = Export(
        True,  # average
        True,  # details
        True,  # convergence
        True,  # boxplot
        True,  # scatter-plot
        True,  # routes
        True,  # configuration
        iterate)

    run(params, export_flags)
