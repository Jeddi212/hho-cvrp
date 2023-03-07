import matplotlib.pyplot as plt
import pandas as pd


def run(results_directory, optimizer, instances, iterations):
    """
    Create a convergence of fitness value of each algorithm

    Parameters
    ----------
    results_directory : str
        directory path the file will be saved
    optimizer :
        list of optimizers
    instances : CVRPLIB CVRP Class
        instance of problem
    iterations :
        number of iterations

    Returns
    -------
    N/A
    """

    plt.ioff()
    file_results_data = pd.read_csv(results_directory + "../experiment_avg.csv")

    for j in range(0, len(instances)):
        instance_name = instances[j]

        start_iteration = 0
        if "SSA" in optimizer:
            start_iteration = 1
        all_generations = [x + 1 for x in range(start_iteration, iterations)]

        for i in range(len(optimizer)):
            optimizer_name = optimizer[i]

            row = file_results_data[
                (file_results_data["Optimizer"] == optimizer_name)
                & (file_results_data["Instance"] == instance_name)
                ]
            row = row.iloc[:, 6 + start_iteration:]
            plt.plot(all_generations, row.values.tolist()[0], label=optimizer_name)
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
        plt.grid()
        fig_name = results_directory + "/convergence-" + instance_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
