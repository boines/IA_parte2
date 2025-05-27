
def load_best_individual_from_file(filename):
    try:
        with open(filename, 'r') as f:
            return list(map(float, f.read().strip().split(',')))
    except FileNotFoundError:
        return None

import random

# cria um indivíduo com valores de pesos entre -1 e 1
def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]


# cria uma população inicial com indivíduos aleatórios
def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]


def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.1, mutation_rate=0.2):
    copies = max(1, int(0.05 * population_size))
    population = generate_population(individual_size, population_size - copies)
    best_from_file = load_best_individual_from_file("best_individual.txt")
    
    if best_from_file:
        for _ in range(copies):
            population.append(best_from_file)
    else:
        for _ in range(copies):
            population.append(create_individual(individual_size))
    best_individual = None
    parent = None
    no_improvement = 0  # contador para estagnação


    for generation in range(generations):
        # avalia o fitness de cada indivíduo
        scored_population = [(ind, fitness_function(ind)) for ind in population]

        # ordena os indivíduos por fitness - maior primeiro
        scored_population.sort(key=lambda x: x[1], reverse=True)

        # guarda melhor indivíduo da geração
        if best_individual is None or scored_population[0][1] > best_individual[1]:
            best_individual = scored_population[0]
            no_improvement = 0  # houve melhoria → reset ao contador
        else:
            no_improvement += 1
        
        print(f"Generation {generation}: Best fitness = {best_individual[1]}")
        
        # se atingiu o fitness alvo termina
        if best_individual[1] >= target_fitness:
            break

        # seleciona os melhores indivíduos 
        elite_size = max(1, int(elite_rate * population_size))
        elites = [ind for ind, _ in scored_population[:elite_size]]

        # geração da nova população
        new_population = elites.copy()

        # se não houve melhoria, incrementa o contador de estagnação
        if no_improvement >= 5:
            print("Injecting diversity...")
            for _ in range(int(0.2 * population_size)):
                parent = random.choice(elites)
                mutated = mutate(parent, mutation_rate * 2)  # mutação mais forte
                new_population.append(mutated)
            #new_individuals = [create_individual(individual_size) for _ in range(int(0.1 * population_size))]
            #new_population += new_individuals
            no_improvement = 0  # reset do contador
        
        while len(new_population) < population_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual # This is expected to be a pair (individual, fitness)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(individual, mutation_rate):
    return [gene + random.uniform(-0.5, 0.5) if random.random() < mutation_rate else gene for gene in individual]