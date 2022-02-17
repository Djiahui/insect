import csv
import pickle

with open('new_temp.csv') as f:
	temp = csv.reader(f)
	rows = [row for row in temp]
	print(rows[0])
	print(rows[1])
temperature_data_new = {}
for row in rows[1:]:
	if row[0] != 'YEAR' and int(row[0])>=2016:
		date = row[0] + '/' + row[1] + '/' + row[2]
		temperature_data_new[date] = float(row[-1])
with open('new_temp_data','wb') as pkl:
	pickle.dump(temperature_data_new,pkl)



