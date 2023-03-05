from optimizer import run


def conf(num, pop, iterate, names):
    # Select optimizers
    optimizer = ["HHO"]

    # Select benchmark function
    instances = names

    # Select number of repetitions for each experiment.
    # To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
    num_of_runs = num

    # Select general parameters for all optimizers (population size, number of iterations) ....
    params = {
        "population_size": pop,
        "iterations": iterate
    }

    # Choose whether to Export the results in different formats
    export_flags = {
        "export_avg": True,
        "export_details": True,
        "export_convergence": True,
        "export_boxplot": True,
        "export_scatter": True,
        "export_route": True,
    }

    run(optimizer, instances, num_of_runs, params, export_flags)
