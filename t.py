import numpy as np
map1 =         ['* * * * * * * * * * * * + + + + + + + +',
			   '* * * * * * * * * * * * + + q + + + + +',
			   '* * * * * * * * * * * * + + + + + + + +',
			   '* * * * * * * * * * * * + + + + + + + +',
			   '* * * * * * * * * * * * + + + + + + + +',
			   '* * * * * * * * * * * * + + + + + + + +',
			   '+ + + + + + + + + + + + + q q + + + + +',
			   '+ + + + + + + + + + + + + + + + + + + +',
			   '+ + + + + + + f f + + f + q q + + + + +',
			   '+ + + + + + + f + + + + + + + + + + + +',
			   '+ + + + m + + f + + + m + q q + + + + +',
			   '+ + + + + + + + + + + + + + + + + + + +',
			   '+ + + + + + + + + f + + + q q + + + + +',
			   '+ + + + + + + + + + + + + + + + + + + +',
			   '+ + + + + f + + + + + + + q q + + + + q',
			   '+ + + + m f + + + + + m + + + + + + f +',
			   '+ + + + + + + + + + + + + + + + + + + q',
			   '+ + + + + + + + + + + + f + + + + + + +',
			   '+ + + + + + + + + + + + + + + + + + + +',
			   '+ + + + + + + + + + + + q q q + + + + +']
temp = [[0 for _ in range(20)] for _ in range(20)]
lists = []
for i in range(20):
	t = map1[i].split(' ')
	for j in range(20):
		if t[j] == 'm':
			temp[i][j] = 0.65
			lists.append((i,j))
		elif t[j] == 'f':
			temp[i][j] = 0.5
			lists.append((i,j))
		elif t[j] == 'q':
			temp[i][j] = 0.55
			lists.append((i,j))

def bfs(index):

	queue = [index]
	dic = [(0,1),(0,-1),(1,0),(-1,0)]
	while queue:
		ii,jj = queue.pop(0)
		for d in dic:
			if 0<=ii+d[0]<=19 and 0<=jj+d[1]<=19 and temp[ii][jj]>=0.1 and temp[ii][jj]-0.1>temp[ii+d[0]][jj+d[1]] and not(ii+d[0]<=5 and jj+d[1]<=11):
				temp[ii+d[0]][jj+d[1]] = temp[ii][jj]-0.1
				queue.append((ii+d[0],jj+d[1]))

for index in lists:
	bfs(index)


def fun(x):
	return round(x,2)
tt = []
for ttt in temp:
	tt.append(list(map(lambda x: round(x,2),ttt)))
t = np.array(tt)

exit()
