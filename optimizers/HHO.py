import time
import random
import math
from optimizers.two_opt import two_opt

from solution import Solution
from benchmarks import split_customer
from benchmarks import concat_depot
from benchmarks import normal_cvrp
from generate import n_vehicle

import numpy


def hho(objf, data, search_agent_no, max_iter):
    lb, ub, dim, distances = 1, data.dimension - 0.01, data.n_customers, data.distances
    max_capacity, demands = data.capacity, data.demands

    # initialize the location and Energy of the rabbit
    rabbit_location = numpy.zeros(dim)
    rabbit_energy = float("inf")  # change this to -inf for maximization problems
    fitness = float("inf")

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)

    # Initialize the locations of Harris' hawks
    x_hawks = numpy.asarray(
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

        # Check boundaries
        x_hawks[i, :] = numpy.clip(x_hawks[i, :], lb, ub)

        # fitness of locations
        x_hawks[i, :] = get_permutation(x_hawks[i, :])
        fitness = objf(x_hawks[i, :].astype(int), distances, max_capacity, demands)

        # Update the location of Rabbit
        if fitness < rabbit_energy:  # Change this to > for maximization problem
            rabbit_energy = fitness
            rabbit_location = x_hawks[i, :].copy()

    # Main loop
    while t < max_iter - 2:
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
                    x_hawks[i, :] = x_rand + levy(dim) * abs(
                        x_hawks.mean(0) - x_hawks[i, :]
                    )

                elif q < 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    x_hawks[i, :] = x_hawks.mean(0) - levy(dim) * (
                            rabbit_location - x_hawks[i, :]
                    )

            # -------- Exploitation phase -------------------
            elif abs(escaping_energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probability of each event

                if (
                        r >= 0.5
                ):  # Hard soft besiege Eq. (6) in paper
                    jump_strength = 2 * (1 - random.random())
                    x_hawks[i, :] = (rabbit_location - x_hawks[i, :]) - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks[i, :]
                    ) - (1 - escaping_energy) * abs(x_hawks[i, :])

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                        r < 0.5
                ):  # Hard Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    jump_strength = 2 * (1 - random.random())
                    x1 = rabbit_location - escaping_energy * levy(dim) * abs(
                        jump_strength * rabbit_location - x_hawks[i, :]
                    )
                    x1 = numpy.clip(x1, lb, ub)
                    x1 = get_permutation(x1)
                    if objf(x1, distances, max_capacity, demands) < fitness:  # improved move?
                        x_hawks[i, :] = x1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        x2 = (
                                rabbit_location
                                - (1 - escaping_energy)
                                * levy(dim) * abs(jump_strength * rabbit_location - x_hawks.mean(0))
                        )
                        x2 = numpy.clip(x2, lb, ub)
                        x2 = get_permutation(x2)
                        if objf(x2, distances, max_capacity, demands) < fitness:
                            x_hawks[i, :] = x2.copy()

        for i in range(0, search_agent_no):

            # Check boundaries
            x_hawks[i, :] = numpy.clip(x_hawks[i, :], lb, ub)

            # fitness of locations
            if t < max_iter - 3:
                x_hawks[i, :] = get_permutation(x_hawks[i, :])
            else:
                x_hawks[i, :] = two_opt(concat_depot(get_permutation(x_hawks[i, :])), distances)[1:-1]
            fitness = objf(x_hawks[i, :].astype(int), distances, max_capacity, demands)

            # Update the location of Rabbit
            if fitness < rabbit_energy:  # Change this to > for maximization problem
                rabbit_energy = fitness
                rabbit_location = x_hawks[i, :].copy()

        convergence_curve[t] = rabbit_energy
        if t % 1 == 0:
            print(
                "At iteration "
                + str(t)
                + " the best fitness is "
                + str(rabbit_energy)
            )
        t = t + 1

    next_best = float("inf")
    the_best = []
    n_v = n_vehicle(data.name)
    for plan in split_array(rabbit_location.astype(int), n_v):
        if all(test_capacity(r, max_capacity, demands) for r in plan):
            temp = normal_cvrp([concat_depot(r) for r in plan], distances)
            if temp < next_best:
                next_best = temp
                the_best = plan

    convergence_curve[t] = next_best
    if t % 1 == 0:
        print(
            "At iteration "
            + str(t)
            + " the best fitness is "
            + str(next_best)
        )
    t += 1

    the_best = [two_opt(ru, distances) for ru in [concat_depot(t) for t in the_best]]
    next_best = normal_cvrp(the_best, distances)

    convergence_curve[t] = next_best
    if t % 1 == 0:
        print(
            "At iteration "
            + str(t)
            + " the best fitness is "
            + str(next_best)
        )
    t += 1

    timer_end = time.time()
    s.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.execution_time = timer_end - timer_start
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = next_best
    s.best_individual = rabbit_location
    s.name = data.name
    # s.routes = split_customer(rabbit_location.astype(int), max_capacity, demands)
    s.routes = the_best
    s.dim = data.dimension
    s.coordinates = data.coordinates

    return s


def levy(dim):
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


def get_permutation(arr):
    """
    This function takes a 1-dimensional list or array as input and returns a list of indices that correspond to the
    sorted elements in the original list or array. The returned list of indices can be used to access the sorted
    elements in the original list or array. The sorting is performed in ascending order based on the values of the
    elements in the list or array.

    :param arr: 1-dimensional array-like
    :return: 1-dimensional array-like

    Examples
    --------
    >>> ary = [0.1, 0.5, 0.3, 0.7, 0.2]
    >>> print(get_permutation(ary))
    [0, 4, 2, 1, 3]

    """
    return [i + 1 for i, x in sorted(enumerate(arr), key=lambda x: x[1])]


def cvrp_two_opt(routes, distances):
    return [two_opt(r, distances) for r in routes]


def cvrp_two_opt_no_depot(routes, distances):
    return [y for r in routes for y in two_opt(r, distances)[1:-1]]


def split_array(arr, k):
    if k == 1:
        yield [arr]
    elif k == len(arr):
        yield [arr[i:i + 1] for i in range(len(arr))]
    else:
        for i in range(1, len(arr)):
            for right in split_array(arr[i:], k - 1):
                yield [arr[:i]] + right


def test_capacity(route, max_capacity, demands):
    return True if sum([demands[c] for c in route]) <= max_capacity else False
