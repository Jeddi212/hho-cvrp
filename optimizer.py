import csv
import time
import warnings
from pathlib import Path

import numpy

import benchmarks
from optimizers.hho_cvrp import hho
import plot_boxplot as box_plot
import plot_convergence as conv_plot
import plot_scatter as scatter_plot
import write_configuration
import write_routes

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


def run(optimizer, instances, num_of_runs, params: dict[str, int], export_flags: dict[str, bool]):
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
    export_flags : set
        The set of Boolean flags which are:
        1. export (Exporting the results in a file)
        2. export_details (Exporting the detailed results in files)
        3. export_convergence (Exporting the convergence plots)
        4. export_boxplot (Exporting the box plots)
        5. export_scatter (Exporting the scatter plots)
        6. export_route (Exporting the routes for each iteration)
        7. export_configuration: (Exporting the configuration of current test)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    population_size = params["population_size"]
    iterations = params["iterations"]

    # export results ?
    export = export_flags["export_avg"]
    export_details = export_flags["export_details"]
    export_convergence = export_flags["export_convergence"]
    export_boxplot = export_flags["export_boxplot"]
    export_scatter = export_flags["export_scatter"]
    export_route = export_flags["export_route"]
    export_configuration = export_flags["export_configuration"]

    flag = False
    flag_details = False

    # CSV Header for the convergence
    cnvg_header = []

    results_directory = "out/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for it in range(0, iterations):
        cnvg_header.append("Iter" + str(it + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(instances)):
            convergence = [0] * num_of_runs
            execution_time = [0] * num_of_runs
            best_solution = [0] * num_of_runs
            gap_solution = [0] * num_of_runs
            for k in range(0, num_of_runs):
                func_details = benchmarks.get_function_details(instances[j])
                x = selector(optimizer[i], func_details, population_size, iterations)
                convergence[k] = x.convergence

                if export_details:
                    export_to_file = results_directory + "experiment_details.csv"
                    with open(export_to_file, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if not flag_details:
                            # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "Instance", "BKS", "BS", "Gap", "ExecutionTime"], cnvg_header]
                            )
                            writer.writerow(header)
                            flag_details = True  # at least one experiment
                        execution_time[k] = x.execution_time
                        best_solution[k] = x.best
                        gap_solution[k] = benchmarks.gap(x.bks, x.best)
                        a = numpy.concatenate(
                            [
                                [
                                    x.optimizer,
                                    x.name,
                                    x.bks,
                                    x.best,
                                    gap_solution[k],
                                    x.execution_time
                                ],
                                x.convergence
                            ]
                        )
                        writer.writerow(a)
                    out.close()

                if export_route:
                    rd = results_directory + "routes-" + x.optimizer + "/" + x.name + "/"
                    Path(rd).mkdir(parents=True, exist_ok=True)
                    write_routes.run(x, rd, k)

                if export_scatter:
                    close = "/" if x.coordinates is not None else "/None"
                    rd = results_directory + "scatter-plot-" + x.optimizer + "/" + x.name + close
                    Path(rd).mkdir(parents=True, exist_ok=True)

                    scatter_plot.run(x, rd, k) if x.coordinates is not None else ()

            if export:
                export_to_file = results_directory + "experiment_avg.csv"

                with open(export_to_file, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if not flag:
                        # just one time to write the header of the CSV file
                        header = numpy.concatenate(
                            [["Optimizer", "Instance", "BKS", "BS", "Gap", "ExecutionTime"], cnvg_header]
                        )
                        writer.writerow(header)
                        flag = True

                    avg_execution_time = float("%0.2f" % (sum(execution_time) / num_of_runs))
                    avg_bs = float("%0.2f" % (sum(best_solution) / num_of_runs))
                    avg_gap = float("%0.4f" % (sum(gap_solution) / num_of_runs))
                    avg_convergence = numpy.around(
                        numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                    ).tolist()
                    a = numpy.concatenate(
                        [
                            [
                                x.optimizer,
                                x.name,
                                x.bks,
                                avg_bs,
                                avg_gap,
                                avg_execution_time
                            ],
                            avg_convergence
                        ]
                    )
                    writer.writerow(a)
                out.close()

    if export_convergence:
        rd = results_directory + "convergence-plot/"
        Path(rd).mkdir(parents=True, exist_ok=True)
        conv_plot.run(rd, optimizer, instances, iterations)

    if export_boxplot:
        rd = results_directory + "box-plot/"
        Path(rd).mkdir(parents=True, exist_ok=True)
        box_plot.run(rd, optimizer, instances, iterations)

    if export_configuration:
        export_to_file = results_directory + "configuration.txt"
        write_configuration.run(export_to_file, num_of_runs, population_size, iterations, instances)

    if not flag:  # Failed to run at least one experiment
        print(
            "No Optimizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
