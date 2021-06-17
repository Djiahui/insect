def fast_nondominated_sort(population):
	population.fronts = [[]]
	for individual in population.pops:
		individual.domination_count = 0
		individual.dominated_solutions = []
		for other_individual in population.pops:
			if individual.dominates(other_individual):
				individual.dominated_solutions.append(other_individual)
			elif other_individual.dominates(individual):
				individual.domination_count += 1
		if individual.domination_count == 0:
			individual.rank = 0
			population.fronts[0].append(individual)
	i = 0
	while len(population.fronts[i]) > 0:
		temp = []
		for individual in population.fronts[i]:
			for other_individual in individual.dominated_solutions:
				other_individual.domination_count -= 1
				if other_individual.domination_count == 0:
					other_individual.rank = i + 1
					temp.append(other_individual)
		i = i + 1
		population.fronts.append(temp)


def calculate_crowding_distance(front):
	if len(front) > 0:
		solutions_num = len(front)
		for individual in front:
			individual.crowding_distance = 0

		for m in range(len(front[0].objectives)):
			front.sort(key=lambda individual: individual.objectives[m])
			front[0].crowding_distance = 10 ** 9
			front[solutions_num - 1].crowding_distance = 10 ** 9
			m_values = [individual.objectives[m] for individual in front]
			scale = max(m_values) - min(m_values)
			if scale == 0: scale = 1
			for i in range(1, solutions_num - 1):
				front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale



if __name__ == '__main__':
	exit()