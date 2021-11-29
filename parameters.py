import random
class Parameters(object):
	prob_m = 0.65
	prob_f = 0.55
	prob_q = 0.5

	insect_num = 13
	x = 200
	y = 200
	step = 10
	sample_num = 180
	insect_iteration = 90
	pop_num = 10
	test = True
	insect_fall_machine = 1
	def get_random_insect_number(self):
		return random.randint(1,500)


if __name__ == '__main__':
	pass
