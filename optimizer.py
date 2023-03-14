import time
import warnings
from pathlib import Path

import benchmarks
from optimizers.hho_cvrp import hho
from model.collection import Collection
from model.export import Export
import plot_boxplot as box_plot
import plot_convergence as conv_plot
import plot_scatter as scatter_plot

warnings.simplefilter(action="ignore")


def selector(algo, func_details, pop_size, n_iter):
    """

    Parameters
    ----------
    algo : list of str
        list of algorithm used to solve the problem
    func_details : list
        Contain name of objective function, instance, and solution
    pop_size : int
        population size for each algorithm
    n_iter : int
        number of iteration for each algorithm

    Returns
    -------
    x : Solution Class
        Solution class that include much information about optimizer process result
    """

    function_name = func_details[0]
    instance = func_details[1]
    solution = func_details[2]

    if algo == "HHO":
        x = hho(getattr(benchmarks, function_name), instance, solution, pop_size, n_iter)
    else:
        return None
    return x


def run(optimizer, instances, num_of_runs, params: dict[str, int], export: Export):
    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    instances : list
        The list of benchmark instances
    num_of_runs : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (population_size)
        2. The number of iterations (iterations)
    export : Export
        The set of Boolean flags which are:
        1. export.avg (Exporting the results in a file)
        2. export.boxplot (Exporting the box plots)
        3. export.configuration: (Exporting the configuration of current test)
        4. export.convergence (Exporting the convergence plots)
        5. export.details (Exporting the detailed results in files)
        6. export.route (Exporting the routes for each iteration)
        7. export.scatter (Exporting the scatter plots)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    population_size = params["population_size"]
    iterations = params["iterations"]

    flag = False
    flag_detail = False

    # CSV Header for the convergence
    cnvg_header = []

    results_directory = "out/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for it in range(0, iterations):
        cnvg_header.append("Iter" + str(it + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(instances)):
            collection = Collection(num_of_runs)
            for k in range(0, num_of_runs):
                func_details = benchmarks.get_function_details(instances[j])
                solution = selector(optimizer[i], func_details, population_size, iterations)
                collection.convergence[k] = solution.convergence

                if export.details:
                    export_to_file = results_directory + "experiment_details.csv"
                    export.write_detail(export_to_file, flag_detail, collection, solution, cnvg_header, k)
                    flag_detail = True

                if export.route:
                    rd = results_directory + "routes-" + solution.optimizer + "/" + solution.name + "/"
                    Path(rd).mkdir(parents=True, exist_ok=True)
                    export.write_route(rd, solution, k)

                if export.scatter:
                    close = "/" if solution.coordinates is not None else "/None"
                    rd = results_directory + "scatter-plot-" + solution.optimizer + "/" + solution.name + close
                    Path(rd).mkdir(parents=True, exist_ok=True)

                    scatter_plot.run(solution, rd, k) if solution.coordinates is not None else ()

            if export.avg:
                export_to_file = results_directory + "experiment_avg.csv"
                export.write_avg(export_to_file, flag, collection, solution, num_of_runs, cnvg_header)
                flag = True

    if export.convergence:
        rd = results_directory + "convergence-plot/"
        Path(rd).mkdir(parents=True, exist_ok=True)
        conv_plot.run(rd, optimizer, instances, iterations)

    if export.boxplot:
        rd = results_directory + "box-plot/"
        Path(rd).mkdir(parents=True, exist_ok=True)
        box_plot.run(rd, optimizer, instances, iterations)

    if export.configuration:
        export_to_file = results_directory + "configuration.txt"
        export.write_configuration(export_to_file, num_of_runs, population_size, iterations, instances)

    if not flag:  # Failed to run at least one experiment
        print(
            "No Optimizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
