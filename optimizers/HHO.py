import time
import random
import math

from optimizers.crossover import pmx
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


def xr(xh):
    return numpy.around(
        xh, decimals=2
    )


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
    fs = [float("inf") for _ in range(search_agent_no)]

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.array(lb)
    ub = numpy.array(ub)

    # Initialize the locations of Harris' hawks
    x_hawks = numpy.array(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (search_agent_no, dim))]
    )

    print("FASE INISIALISASI POPULASI AWAL")
    x_hawks = xr(x_hawks)

    for idx, z in enumerate(x_hawks):
        print(f"Elang {idx} - {list(z)}")
    print()

    # Initialize convergence
    convergence_curve = numpy.zeros(max_iter)

    ############################
    s = Solution()

    print('HHO is now tackling "' + objf.__name__ + '" ' + data.name)

    timer_start = time.time()
    s.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    print("TENTUKAN KELINCI AWAL & RANDOM KEY")
    for i in range(0, search_agent_no):

        # fitness of locations
        x_hawks[i, :] = random_key(x_hawks[i, :])
        print(f"ELANG {i} - {list(x_hawks[i, :].astype(int))}")
        fitness = objf(x_hawks[i, :].astype(int), distances, max_capacity, demands)
        print(f"Fitness : {fitness}")
        fs[i] = fitness

        # Update the location of Rabbit
        if fitness < rabbit_energy:  # Change this to > for maximization problem
            rabbit_energy = fitness
            rabbit_location = x_hawks[i, :].copy()
    print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")
    print(f"Rabbit : {rabbit_energy}")
    print()
    print()
    print()
    print()
    print()

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

            match t:
                case 0 | 1:
                    escaping_energy = 1.5
                case 2 | 4 | 6:
                    escaping_energy = 0.6
                case 3 | 5 | 7:
                    escaping_energy = 0.1

            if abs(escaping_energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_hawk_index = math.floor(search_agent_no * random.random())
                x_rand = x_hawks[rand_hawk_index, :]

                match t:
                    case 0:
                        q = 0.9
                    case 1:
                        q = 0.1

                if q >= 0.5:
                    print(f"MASUK Exploration 1 - E: {escaping_energy}, q: {q}")
                    print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                    print(f"RANDM {rand_hawk_index} -> {list(x_rand)}")
                    # perch based on other family members
                    r1 = round(random.random(), 2)
                    r2 = round(random.random(), 2)
                    print(f"r1: {r1}, r2: {r2}")
                    x_hawks[i, :] = x_rand - r1 * abs(
                        x_rand - 2 * r2 * x_hawks[i, :]
                    )
                    x_hawks[i, :] = xr(x_hawks[i, :])
                    print(f"ELANG 2.9  {i} -> {list(x_hawks[i, :])}")
                    x_hawks[i, :] = random_key(x_hawks[i, :])
                    print(f"ELANG RK   {i} -> {list(x_hawks[i, :].astype(int))}")
                    x_hawks[i, :] = mutate.swap(x_hawks[i, :])
                    print(f"ELANG swap {i} -> {list(x_hawks[i, :].astype(int))}")

                elif q < 0.5:
                    print(f"MASUK Exploration 2 - E: {escaping_energy}, q: {q}")
                    print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                    x_mean = xr(x_hawks.mean(0))
                    print(f"MEAN    -> {list(x_mean)}")
                    r3 = round(random.random(), 2)
                    r4 = round(random.random(), 2)
                    print(f"r3: {r3}, r4: {r4}")

                    # perch on a random tall tree (random site inside group's home range)
                    x_hawks[i, :] = (rabbit_location - x_mean) - r3 * (
                            (ub - lb) * r4 + lb
                    )
                    x_hawks[i, :] = xr(x_hawks[i, :])
                    print(f"ELANG 2.9  {i} -> {list(x_hawks[i, :])}")
                    x_hawks[i, :] = random_key(x_hawks[i, :])
                    print(f"ELANG RK   {i} -> {list(x_hawks[i, :].astype(int))}")
                    x_hawks[i, :] = mutate.inverse(x_hawks[i, :])
                    print(f"ELANG invr {i} -> {list(x_hawks[i, :].astype(int))}")

                print()

            # -------- Exploitation phase -------------------
            elif abs(escaping_energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probability of each event

                match t:
                    case 2 | 3:
                        r = 0.7
                    case 4 | 5 | 6 | 7:
                        r = 0.2

                if (
                        r >= 0.5 > abs(escaping_energy)
                ):  # Hard besiege Eq. (6) in paper
                    print(f"MASUK Exploitation 2 - E: {escaping_energy}, r: {r}")
                    print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                    print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")

                    x_hawks[i, :] = rabbit_location - escaping_energy * abs(
                        rabbit_location - x_hawks[i, :]
                    )
                    x_hawks[i, :] = xr(x_hawks[i, :])
                    print(f"ELANG 2.15 {i} -> {list(x_hawks[i, :])}")
                    x_hawks[i, :] = random_key(x_hawks[i, :])
                    print(f"ELANG RK   {i} -> {list(x_hawks[i, :].astype(int))}")
                    x_hawks[i, :] = mutate.swap(x_hawks[i, :])
                    print(f"ELANG swap {i} -> {list(x_hawks[i, :].astype(int))}")

                if (
                        r >= 0.5 and abs(escaping_energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    print(f"MASUK Exploitation 1 - E: {escaping_energy}, r: {r}")
                    print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                    print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")

                    r5 = round(random.random(), 2)
                    print(f"r5: {r5}")
                    jump_strength = round(2 * (
                            1 - r5
                    ), 2)  # random jump strength of the rabbit
                    print(f"Jump Strength: {jump_strength}")

                    x_hawks[i, :] = (rabbit_location - x_hawks[i, :]) - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks[i, :]
                    )
                    x_hawks[i, :] = xr(x_hawks[i, :])
                    print(f"ELANG 2.12 {i} -> {list(x_hawks[i, :])}")
                    x_hawks[i, :] = random_key(x_hawks[i, :])
                    print(f"ELANG RK   {i} -> {list(x_hawks[i, :].astype(int))}")
                    x_hawks[i, :] = mutate.swap(x_hawks[i, :])
                    print(f"ELANG swap {i} -> {list(x_hawks[i, :].astype(int))}")

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                        r < 0.5 <= abs(escaping_energy)
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    print(f"MASUK Exploitation WPRD 1 - E: {escaping_energy}, r: {r}")
                    print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                    print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")

                    r5 = round(random.random(), 2)
                    print(f"r5: {r5}")
                    jump_strength = round(2 * (1 - r5), 2)
                    print(f"Jump Strength: {jump_strength}")

                    x1 = rabbit_location - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks[i, :]
                    )
                    x1 = xr(x1)
                    print(f"Y 2.19 {i} -> {list(x1)}")
                    x1 = random_key(x1)
                    print(f"Y RK   {i} -> {list(x1)}")
                    x1 = mutate.swap(x1)
                    print(f"Y swap {i} -> {list(x1)}")

                    y1, y2 = pmx(
                        random_key(rabbit_location),
                        random_key(x1)
                    )
                    print(f"Y PMX1  {i} -> {list(y1)}")
                    print(f"Y PMX1  {i} -> {list(y2)}")
                    yobjf1 = objf(y1, distances, max_capacity, demands)
                    yobjf2 = objf(y2, distances, max_capacity, demands)
                    print(f"Y objf y1: {yobjf1}")
                    print(f"Y objf y2: {yobjf2}")

                    Y = y1 if yobjf1 < yobjf2 else y2
                    print(f"MAKA Y -> {Y}")

                    if objf(Y, distances, max_capacity, demands) < fs[i]:  # improved move?
                        x_hawks[i, :] = Y.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        print(f"MASUK Exploitation WPRD 2 - E: {escaping_energy}, r: {r}")
                        print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                        print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")

                        S = numpy.round(numpy.random.randn(dim), 4)
                        print(f"S    -> {list(S)}")
                        LF = numpy.round(levy(dim), 4)
                        print(f"LF    -> {list(LF)}")
                        x2 = (
                                rabbit_location
                                - escaping_energy
                                * abs(jump_strength * rabbit_location - x_hawks[i, :])
                                + numpy.multiply(S, LF)
                        )
                        x2 = xr(x2)
                        print(f"Z 2.19 {i} -> {list(x2)}")
                        x2 = random_key(x2)
                        print(f"Z RK   {i} -> {list(x2)}")
                        x2 = mutate.insertion(x2)
                        print(f"Z isrt {i} -> {list(x2)}")
                        x2 = two_opt_inverse(concat_depot(random_key(x2)), distances)[1:-1]
                        print(f"Z 2opt {i} -> {list(x2)}")

                        zobjf = objf(x2, distances, max_capacity, demands)
                        print(f"Z objf {i}: {zobjf}")

                        if zobjf < fs[i]:
                            x_hawks[i, :] = x2.copy()
                if (
                        r < 0.5 and abs(escaping_energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    print(f"MASUK Exploitation WPRD 3 - E: {escaping_energy}, r: {r}")
                    print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                    print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")
                    x_mean = xr(x_hawks.mean(0))
                    print(f"MEAN    -> {list(x_mean)}")
                    r5 = round(random.random(), 2)
                    print(f"r5: {r5}")

                    jump_strength = round(2 * (1 - r5), 2)
                    print(f"Jump Strength: {jump_strength}")

                    x1 = rabbit_location - escaping_energy * abs(
                        jump_strength * rabbit_location - x_hawks.mean(0)
                    )
                    x1 = xr(x1)
                    print(f"Y 2.19 {i} -> {list(x1)}")
                    x1 = random_key(x1)
                    print(f"Y RK   {i} -> {list(x1)}")
                    x1 = mutate.inverse(x1)
                    print(f"Y invr {i} -> {list(x1)}")

                    yobjf = objf(x1, distances, max_capacity, demands)
                    print(f"Y objf {i}: {yobjf}")

                    if yobjf < fs[i]:  # improved move?
                        x_hawks[i, :] = x1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        print(f"MASUK Exploitation WPRD 4 - E: {escaping_energy}, r: {r}")
                        print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")
                        print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")
                        x_mean = xr(x_hawks.mean(0))
                        print(f"MEAN    -> {list(x_mean)}")

                        S = numpy.round(numpy.random.randn(dim), 4)
                        print(f"S    -> {list(S)}")
                        LF = numpy.round(levy(dim), 4)
                        print(f"LF    -> {list(LF)}")
                        x2 = (
                                rabbit_location
                                - escaping_energy
                                * abs(jump_strength * rabbit_location - x_hawks.mean(0))
                                + numpy.multiply(S, LF)
                        )
                        x2 = xr(x2)
                        print(f"Z 2.19 {i} -> {list(x2)}")
                        x2 = random_key(x2)
                        print(f"Z RK   {i} -> {list(x2)}")
                        x2 = mutate.insertion(random_key(x2))
                        print(f"Z isrt {i} -> {list(x2)}")

                        zobjf = objf(x2, distances, max_capacity, demands)
                        print(f"Z objf {i}: {zobjf}")

                        if zobjf < fs[i]:
                            x_hawks[i, :] = x2.copy()
                print()

        print()

        print(f"TENTUKAN KELINCI LAGI PADA ITER {t}")
        for i in range(0, search_agent_no):
            print(f"ELANG {i} -> {list(x_hawks[i, :].astype(int))}")

            # fitness of locations
            if t < max_iter - 1:
                x_hawks[i, :] = random_key(x_hawks[i, :])
            else:
                print("MASUK finishing phase")

                test_route = split_customer(x_hawks[i, :].astype(int), max_capacity, demands)
                print(f"SPLIT {i} -> {test_route}")

                test_route = cvrp_inverse(test_route, distances)
                print(f"CVRP INVERSE {i} -> {test_route}")

                test_route = cvrp_insertion(test_route, distances)
                print(f"CVRP INSERTION {i} -> {test_route}")

            fitness = objf(x_hawks[i, :].astype(int), distances, max_capacity, demands
                           ) if t < max_iter - 1 else normal_cvrp(test_route, distances)
            print(f"Fitness : {fitness}")
            fs[i] = fitness

            # Update the location of Rabbit
            if fitness < rabbit_energy:  # Change this to > for maximization problem
                rabbit_energy = fitness
                rabbit_location = x_hawks[i, :].copy()

                if t == max_iter - 1:
                    best_route = test_route
        print(f"Rabbit Location -> {list(rabbit_location.astype(int))}")
        print(f"Rabbit : {rabbit_energy}")

        convergence_curve[t] = rabbit_energy
        if t % 1 == 0:
            print(
                "At iteration "
                + str(t)
                + " the best fitness is "
                + str(rabbit_energy)
            )
        t = t + 1

        print()
        print()
        print()
        print()
        print()

        if rabbit_energy <= bks:
            convergence_curve = [rabbit_energy if conv == 0 else conv for conv in convergence_curve]
            break

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
