import random
class Parameters(object):
	prob_m = 0.65
	prob_f = 0.55
	prob_q = 0.5

	insect_num = 100
	insect_iteration = 180
	scenario_num = 100

	x = 200
	y = 200
	step = 10
	sample_num = 180

	iteration = 5
	pop_num = 50
	test = True
	insect_fall_machine = 1


	eliminated_number = 3*pop_num-5
	archive_maximum = 50
	alpha = 0.8

	max_insect_num = 100
	min_insect_num = 13

	season = None
	Treatment = True
	@classmethod
	def get_random_insect_number(self):
		return random.randint(13,100)


	threshold = 0.6
	discount_q = 0.2
	discount_p = 0.3

	P = 2
	Q = 5000


if __name__ == '__main__':
	pass
