import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import interpolate

percent = [0, 10,20,30,40,50,60, 65, 67,70,75, 80,85,90,91,92,93,94,95,96,97,98,99 ,100]
viscosity = [1.005, 1.31, 1.76, 2.60, 3.72, 6.00, 10.8, 15.2, 17.7, 22.5, 35.5, 60.1, 109, 219, 259, 310, 367, 437, 523, 624, 765, 939, 1150, 1412]

interpolated_viscosities = [82.5, 86, 87, 88, 89, 90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5]

plt.scatter(percent, viscosity)
#create interpolation for the data


x = np.array(percent)
y = np.array(viscosity)

f = interpolate.interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 100, num=1000, endpoint=True)
ynew = f(xnew)
# interpolated_viscosities = f(np.array(interpolated_viscosities))
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.plot(interpolated_viscosities, f(np.array(interpolated_viscosities)), 'x')
# plt.plot(x, y, 'o', xnew, ynew, '-')
plt.xlabel('Glycogen (Wt %)')
plt.ylabel('Viscosity (cP)')
plt.grid(True)
# plt.show()

print(f"Interpolated viscosities: {f(interpolated_viscosities)}")

percent_all = percent + interpolated_viscosities
percent_all.sort()

viscosity_all = viscosity + f(interpolated_viscosities).tolist()
viscosity_all.sort()
dic = {}


for percent, viscosity in zip(percent_all, viscosity_all):

    dic["P" + str(percent)] = round(viscosity, 2)

print(dic)


    
    



# Change the directory to the specific folder
# path = '/content/drive/My Drive/Regression_Dataset/rgb'

# path = "fokder with new data"



# for file in os.listdir(path):
#   old_label = file.split('_', 1)
#   os.rename(file, dic[old_label[0]] + '_' + old_label[1])