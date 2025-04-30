import random
import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from deap import base, creator, tools

from moead import MOEAD
from real_data import fetch_stock_data, get_esg_scores

# --- Parameters ---
stocks = [
    '0005.HK', '1299.HK', '0939.HK', '1398.HK', '0388.HK', '3988.HK', '2318.HK', '3968.HK', '2388.HK', '2628.HK',
    '0011.HK', '0700.HK', '9988.HK', '3690.HK', '1810.HK', '9618.HK', '9999.HK', '0992.HK', '9888.HK', '0981.HK',
    '0285.HK', '1211.HK', '0669.HK', '2015.HK', '2020.HK', '9633.HK', '0027.HK', '6690.HK', '9961.HK', '1876.HK',
    '2313.HK', '0175.HK', '0291.HK', '1928.HK', '0066.HK', '2319.HK', '2331.HK', '0288.HK', '6862.HK', '1929.HK',
    '0322.HK', '1044.HK', '0881.HK', '0016.HK', '1109.HK', '0823.HK', '1113.HK', '0688.HK', '1997.HK', '0012.HK',
    '0960.HK', '1209.HK', '0101.HK', '0017.HK', '0941.HK', '0002.HK', '0003.HK', '0006.HK', '2688.HK', '0836.HK',
    '0762.HK', '1038.HK', '1093.HK', '2269.HK', '1177.HK', '6618.HK', '1099.HK', '3692.HK', '0241.HK', '2359.HK',
    '0883.HK', '0857.HK', '0386.HK'
]

start_date = datetime.datetime(2023, 3, 1)
end_date = datetime.datetime(2024, 9, 30)

# Fetch real data
expected_returns, cov_matrix, valid_stocks = fetch_stock_data(stocks, start_date, end_date)
esg_scores = np.array(list(map(get_esg_scores, valid_stocks)))

N_STOCKS = len(valid_stocks)
NGEN = 100
MU = 100
LAMBDA = 2
CXPB = 0.9
MUTPB = 0.2

# --- Evaluation function ---
def eval_portfolio(individual):
    weights = np.array(individual)
    weights /= np.sum(weights)

    exp_return = np.dot(weights, expected_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    esg = np.dot(weights, esg_scores)

    return exp_return, -volatility, esg

# --- Crossover and Mutation ---
def cx_blend(ind1, ind2, alpha=0.5):
    for i in range(len(ind1)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i], ind2[i] = (1. - gamma) * ind1[i] + gamma * ind2[i], gamma * ind1[i] + (1. - gamma) * ind2[i]
    return ind1, ind2

def mut_gaussian(individual, mu=0, sigma=0.05, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            individual[i] = min(max(individual[i], 0.0), 1.0)
    return individual,

# --- Main Execution ---
# def main(seed=64):
#     random.seed(seed)
#     np.random.seed(seed)

#     creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, 1.0))
#     creator.create("Individual", list, fitness=creator.Fitness)

#     toolbox = base.Toolbox()
#     toolbox.register("attr_weight", random.random)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=N_STOCKS)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#     toolbox.register("evaluate", eval_portfolio)
#     toolbox.register("mate", cx_blend)
#     toolbox.register("mutate", mut_gaussian)
#     toolbox.register("select", tools.selNSGA2)

#     pop = toolbox.population(n=MU)
#     hof = tools.ParetoFront()

#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean, axis=0)
#     stats.register("std", np.std, axis=0)
#     stats.register("min", np.min, axis=0)
#     stats.register("max", np.max, axis=0)

#     moead = MOEAD(pop, toolbox, MU, CXPB, MUTPB, ngen=NGEN, stats=stats, halloffame=hof, nr=LAMBDA)
#     final_pop = moead.execute()

#     return final_pop, stats, hof
def main(seed=64):
    random.seed(seed)
    np.random.seed(seed)

    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_weight", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=N_STOCKS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_portfolio)
    toolbox.register("mate", cx_blend)
    toolbox.register("mutate", mut_gaussian)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    moead = MOEAD(pop, toolbox, MU, CXPB, MUTPB, ngen=NGEN, stats=stats, halloffame=hof, nr=LAMBDA)
    final_pop = moead.execute()

    # Ensure valid_stocks is calculated and returned.
    # Valid stock names can be retrieved here, assuming valid_stocks is the list of stock tickers you used.
    valid_stocks = stocks  # You should have this variable already populated somewhere in your code

    return final_pop, stats, hof, valid_stocks

def plot_final_population(population):
    returns = [ind.fitness.values[0] for ind in population]
    volatility = [-ind.fitness.values[1] for ind in population]
    esg_scores = [ind.fitness.values[2] for ind in population]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(returns, volatility, esg_scores, c='blue', marker='o')

    ax.set_xlabel('Expected Return')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('ESG Score')
    ax.set_title('Final Population (3D Pareto Front)')

    plt.show()

def calculate_spacing(solutions):
    sorted_solutions = solutions[np.argsort(solutions[:, 0])]
    distances = np.linalg.norm(np.diff(sorted_solutions, axis=0), axis=1)
    average_distance = np.mean(distances)
    return np.mean(np.abs(distances - average_distance))

if __name__ == "__main__":
    seed = 42
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])

    pop, stats, hof = main(seed)

    fitness_values = np.array([ind.fitness.values for ind in pop])

    print("\nFinal Population Fitness Values:")
    for fit in fitness_values:
        print(fit)

    plot_final_population(pop)

    print("\nPareto Front:")
    for ind in hof:
        print(ind.fitness.values)

    worst = np.max(fitness_values, axis=0)
    reference_point = worst + 0.1
    hv = HV(ref_point=reference_point)(fitness_values)

    print(f"\nHypervolume: {hv:.4f}")

    spacing = calculate_spacing(fitness_values)
    print(f"Spacing Metric: {spacing:.4f}")
