import time
import random
import math

from optimizers.encode import random_key
from optimizers.local import two_opt_inverse
from optimizers.local import two_opt_insertion
from optimizers.local import two_opt_swap
import optimizers.mutate as mutate

from solution import Solution
from benchmarks import split_customer
from benchmarks import concat_depot
from benchmarks import normal_cvrp

import numpy


def hho(objf, data, sol, search_agent_no, max_iter):
    """
    This function is Harris Hawks Optimization for CVRP.

    :param objf: an objective function (check benchmarks.py)
    :param data: an instance class downloaded from cvrplib
    :param sol: a solution class downloaded from cvrplib
    :param search_agent_no: number of hawks
    :param max_iter: maximum iteration before it stopped
    :return:
    """
    lb, ub, dim, distances = 1, data.n_customers, data.n_customers, data.distances
    max_capacity, demands = data.capacity, data.demands
    best_route, bks = None, sol.cost

    # initialize the location and Energy of the rabbit
    rabbit_location = numpy.zeros(dim)
    rabbit_energy = float("inf")  # change this to -inf for maximization problems
    fitness = float("inf")

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.array(lb)
    ub = numpy.array(ub)

    # Initialize the locations of Harris' hawks
    x_hawks = numpy.array(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (search_agent_no, dim))]
    )

    # Initialize convergence
    convergence_curve = numpy.zeros(max_iter)

    ############################
    s = Solution()

    print('HHO is now tackling "' + objf.__name__ + '" ' + data.name)

    timer_start = time.time()
    s.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    for i in range(0, search_agent_no):

        # fitness of locations
        x_hawks[i, :] = random_key(x_hawks[i, :])
        fitness = objf(x_hawks[i, :].astype(int), distances, max_capacity, demands)

        # Update the location of Rabbit
        if fitness < rabbit_energy:  # Change this to > for maximization problem
            rabbit_energy = fitness
            rabbit_location = x_hawks[i, :].copy()

    # Main loop
    while t < max_iter:
        e1 = 2 * (1 - (t / max_iter))  # factor to show the decreasing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, search_agent_no):

            e0 = 2 * random.random() - 1  # -1 < e0 < 1
            escaping_energy = e1 * (
                e0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(escaping_energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_hawk_index = math.floor(search_agent_no * random.random())
                x_rand = x_hawks[rand_hawk_index, :]
                if q >= 0.5:
                    # perch based on other family members
                    x_hawks[i, :] = x_rand - random.random() * abs(
                        x_rand - 2 * random.random() * x_hawks[i, :]
                    )
                    x_hawks[i, :] = mutate.swap(random_key(x_hawks[i, :]))

                elif q < 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    x_hawks[i, :] = (rabbit_location - x_hawks.mean(0)) - random.random() * (
                            (ub - lb) * random.random() + lb
                    )
                    x_hawks[i, :] = two_opt_inverse(concat_depot(random_key(x_hawks[i, :])), distances)[1:-1]
                    x_hawks[i, :] = mutate.inverse(random_key(x_hawks[i, :]))

            # -------- Exploitation phase -------------------
            elif abs(escaping_energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probability of each event

                if (
                        r >= 0.5 > abs(escaping_energy)
                ):  # Hard besiege Eq. (6) in paper
                    x_hawks[i, :] = rabbit_location - escaping_energy * abs(
                        rabbit_location - x_hawks[i, :]
                    )
                    x_hawks[i, :] = mutate.swap(random_key(x_hawks[i, :]))

                if (
                        r >= 0.5 and abs(escaping_energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    jump_strength = 2 * (
                            1 - random.random()
                    )  # random jump strength of the rabbit
                    x_hawks[i, :] = (rabbit_location - x_hawks[i, :]) - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks[i, :]
                    )
                    x_hawks[i, :] = mutate.swap(random_key(x_hawks[i, :]))

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                        r < 0.5 <= abs(escaping_energy)
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    jump_strength = 2 * (1 - random.random())
                    x1 = rabbit_location - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks[i, :]
                    )
                    x1 = mutate.swap(random_key(x1))

                    if objf(x1, distances, max_capacity, demands) < fitness:  # improved move?
                        x_hawks[i, :] = x1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        x2 = (
                                rabbit_location
                                - escaping_energy
                                * abs(jump_strength * rabbit_location - x_hawks[i, :])
                                + numpy.multiply(numpy.random.randn(dim), levy(dim))
                        )
                        x2 = mutate.insertion(random_key(x2))
                        # x2 = two_opt_inverse(concat_depot(random_key(x2)), distances)[1:-1]

                        if objf(x2, distances, max_capacity, demands) < fitness:
                            x_hawks[i, :] = x2.copy()
                if (
                        r < 0.5 and abs(escaping_energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    jump_strength = 2 * (1 - random.random())
                    x1 = rabbit_location - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks.mean(0)
                    )
                    x1 = mutate.inverse(random_key(x1))

                    if objf(x1, distances, max_capacity, demands) < fitness:  # improved move?
                        x_hawks[i, :] = x1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        x2 = (
                                rabbit_location
                                - escaping_energy
                                * abs(jump_strength * rabbit_location - x_hawks.mean(0))
                                + numpy.multiply(numpy.random.randn(dim), levy(dim))
                        )
                        x2 = mutate.insertion(random_key(x2))

                        if objf(x2, distances, max_capacity, demands) < fitness:
                            x_hawks[i, :] = x2.copy()

        for i in range(0, search_agent_no):

            # fitness of locations
            if t < max_iter - 1:
                x_hawks[i, :] = random_key(x_hawks[i, :])
            else:
                # if rabbit_energy > bks:
                #     x_hawks[i, :] = two_opt_inverse(
                #         concat_depot(random_key(x_hawks[i, :])), distances
                #     )[1:-1]

                test_route = split_customer(x_hawks[i, :].astype(int), max_capacity, demands)
                test_route = cvrp_inverse(test_route, distances)
                test_route = cvrp_insertion(test_route, distances)

            fitness = objf(x_hawks[i, :].astype(int), distances, max_capacity, demands
                           ) if t < max_iter - 1 else normal_cvrp(test_route, distances)

            # Update the location of Rabbit
            if fitness < rabbit_energy:  # Change this to > for maximization problem
                rabbit_energy = fitness
                rabbit_location = x_hawks[i, :].copy()

                if t == max_iter - 1:
                    best_route = test_route

        convergence_curve[t] = rabbit_energy
        if t % 1 == 0:
            print(
                "At iteration "
                + str(t)
                + " the best fitness is "
                + str(rabbit_energy)
            )
        t = t + 1

    timer_end = time.time()
    s.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.execution_time = timer_end - timer_start
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = rabbit_energy
    s.best_individual = rabbit_location
    s.name = data.name
    s.routes = best_route if best_route is not None else split_customer(
        rabbit_location.astype(int), max_capacity, demands)
    s.dim = data.dimension
    s.coordinates = data.coordinates

    return s


def levy(dim):
    """
    This function is a Lévy flight function.

    :param dim: integer of problem dimension
    :return: value of Lévy flight
    """
    beta = 1.5
    sigma = (
                    math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
                    / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step


def cvrp_insertion(routes, distances):
    """
    This function accepts a list of routes in which
    each route will be processed using insertion 2-opt.

    :param routes: list of routes
    :param distances: distance matrix
    :return: list of routes after insertion 2-opt
    """
    return [two_opt_insertion(r, distances) for r in routes]


def cvrp_inverse(routes, distances):
    """
    This function accepts a list of routes in which
    each route will be processed using inverse 2-opt.

    :param routes: list of routes
    :param distances: distance matrix
    :return: list of routes after inverse 2-opt
    """
    return [two_opt_inverse(r, distances) for r in routes]


def cvrp_swap(routes, distances):
    """
    This function accepts a list of routes in which
    each route will be processed using swap 2-opt.

    :param routes: list of routes
    :param distances: distance matrix
    :return: list of routes after swap 2-opt
    """
    return [two_opt_swap(r, distances) for r in routes]
