import numpy
import cvrplib


def get_function_details(a):
    # Download instances
    instance = cvrplib.download(a)

    # [name, lb, ub, dim]
    param = {
        # dimensi = banyaknya customer
        "cvrp": ["cvrp", instance],
    }
    return param.get("cvrp", "nothing")


def cvrp(solution, distances, max_capacity, demand):
    """
    CVRP objective function sum all distance of routes.
    This function take tsp solution, convert it into cvrp solution,
    then calculate its total distance (fitness value)

    :param solution: 1D list of tsp solution representation
    :param distances: matrix of distances
    :param max_capacity: maximum capacity of truck (homogeneous)
    :param demand: matrix od demands
    :return: the fitness value of cvrp (total distance)

    room for improvement : using built in sum() function instead loops
    """
    routes = split_customer(solution, max_capacity, demand)

    total, start = 0, 0
    for r in routes:
        for i in range(len(r) - 1):
            total += distances[r[i]][r[i + 1]]

    return total


def split_customer(solution, max_capacity, demand):
    """
    This function split tsp solution into cvrp solution

    :param solution: 1D list of tsp solution representation
    :param max_capacity: maximum capacity of truck (homogeneous)
    :param demand: matrix demands
    :return: Lists of route,
             where regular customers are sorted from left -> right

    room for improvement: use numpy
    """
    routes, load, v = [[0]], 0, 0

    for i in solution:
        if demand[i] + load <= max_capacity:
            routes[v].append(i)
            load += demand[i]
        else:
            routes[v].append(0)  # close the route
            routes.append([0, i])  # open new route
            load = demand[i]
            v += 1

    return routes


def concat_depot(s):
    return numpy.concatenate((
        numpy.zeros(1, dtype=int), s, numpy.zeros(1, dtype=int)
    ))
