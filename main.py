import random

from PIL import Image, ImageDraw
from evol import Population, Evolution, Individual

import numpy as np

SIDES = 3
POLYGON_COUNT = 100


def initialize():
    return [make_polygon() for i in range(POLYGON_COUNT)]


def mutate(solution):
    if random.random() < 0.5:
        # mutate points
        polygon = random.choice(solution)
        coords = [x for point in polygon[1:] for x in point]

        import statistics
        from itertools import repeat
        mu = statistics.fmean(coords)
        sd = statistics.stdev(coords, mu)

        size = len(coords)

        for i in range(size):
            if random.random() < 0.5:
                coords[i] += random.gauss(mu, sd)

        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))

    else:
        random.shuffle(solution)
    return solution


def select(population):
    return [random.choice(population) for i in range(2)]


def tournament_selection(population):
    sort = sorted(population, key=lambda x: x.fitness, reverse=True)

    return sort[:2]


def combine(*parents):
    d = []

    for a, b in zip(*parents):
        cp = np.random.randint(1, len(a) - 1)

        # Cut parent1 DNA in halves
        dna_11 = a[:cp]
        dna_12 = a[cp:]

        dna_21 = b[:cp]
        dna_22 = b[cp:]

        # merge the pieces of dna
        child1 = dna_11 + dna_22
        child2 = dna_21 + dna_12

        if random.random() < 0.5:
            d.append(child1)
        else:
            d.append(child2)
        # # mutate childrens' dna with certain probability
        # if np.random.uniform(0, 1) < pmut:
        #     child1 = mutation(child1, size, mtype)
        #
        # if np.random.uniform(0, 1) < pmut:
        #     child2 = mutation(child2, size, mtype)

    return d

def make_polygon():
    x1 = random.randrange(10,190)
    y1 = random.randrange(10, 190)
    x2 = random.randrange(10, 190)
    y2 = random.randrange(10, 190)
    x3 = random.randrange(10, 190)
    y3 = random.randrange(10, 190)
    R = random.randrange(0,256)
    G = random.randrange(0, 256)
    B = random.randrange(0, 256)
    A = random.randrange(30, 61)

    return [(R, G, B, A), (x1, y1), (x2, y2), (x3, y3)]


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])
    return image


from PIL import ImageChops

MAX = 255 * 200 * 200
TARGET = Image.open(r"C:\Users\Josh\Downloads\darwin.png")
TARGET.load()  # read image and close the file


def evaluate(solution):
    image = draw(solution)
    diff = np.array(ImageChops.difference(image, TARGET).getdata()).sum()

    white = Image.new('RGB', TARGET.size, (255, 255, 255))
    mdiff = ImageChops.difference(white, TARGET)
    maxdiff = np.array(mdiff.getdata()).sum()

    return (1 - diff / maxdiff) * 100


def given_evaluate(solution):
    image = draw(solution)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX

def run(generations=1000, population_size=100, seed=30):
    random.seed(seed)

    population = Population.generate(initialize, given_evaluate, size=population_size, maximize=True)
    population.evaluate()

    evolution = (Evolution().survive(fraction=0.5)
                 .breed(parent_picker=tournament_selection, combiner=combine)
                 .mutate(mutate_function=mutate)
                 .evaluate())

    stuff = []

    for i in range(generations):
        population = population.evolve(evolution)
        stuff.append(population.current_best.fitness)

        if i % 20 == 0 or i == generations-1:
            draw(population[len(population) - 1].chromosome).save("solution %i.png" % i)

            print("i =", i, " best =", population.current_best.fitness,
                  " worst =", population.current_worst.fitness)

    import matplotlib.pyplot as plt
    plt.plot(
        range(generations),
        stuff
    )
    plt.title('Some Title')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig('fitness_plot.png')

if __name__ == "__main__":
    run()
