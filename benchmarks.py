import numpy
import cvrplib


def get_function_details(a):
    # Download instances
    instance, solution = cvrplib.download(a, solution=True)

    # [name, instance, solution]
    param = {
        # dimensi = banyaknya customer
        "cvrp": ["cvrp", instance, solution],
    }
    return param.get("cvrp", "nothing")


def cvrp(solution, distances, max_capacity, demands):
    """
    CVRP objective function sum all distance of routes.
    This function take tsp solution, convert it into cvrp solution,
    then calculate its total distance (fitness value)

    :param solution: 1D list of tsp solution representation
    :param distances: matrix of distances
    :param max_capacity: maximum capacity of truck (homogeneous)
    :param demands: matrix od demands
    :return: the fitness value of cvrp (total distance)

    room for improvement : using built in sum() function instead loops
    """
    routes = split_customer(solution, max_capacity, demands)

    print(routes)

    total = 0
    for r in routes:
        for i in range(len(r) - 1):
            total += distances[r[i]][r[i + 1]]

    return total


def normal_cvrp(solution, distances):
    total = 0
    for r in solution:
        for i in range(len(r) - 1):
            total += distances[r[i]][r[i + 1]]

    return total


def split_customer(solution, max_capacity, demands):
    """
    This function split tsp solution into cvrp solution

    :param solution: 1D list of tsp solution representation
    :param max_capacity: maximum capacity of truck (homogeneous)
    :param demands: matrix demands
    :return: Lists of route,
             where regular customers are sorted from left -> right

    room for improvement: use numpy
    """
    routes, load, v = [[0]], 0, 0

    for i in solution:
        if demands[i] + load <= max_capacity:
            routes[v].append(i)
            load += demands[i]
        else:
            routes[v].append(0)  # close the route
            routes.append([0, i])  # open new route
            load = demands[i]
            v += 1

    routes[v].append(0)  # close the last route
    return routes


def concat_depot(s):
    return numpy.concatenate((
        numpy.zeros(1, dtype=int), s, numpy.zeros(1, dtype=int)
    ))
