
from plotting_environment import linear_interpolation

T_air = map(lambda x: x+ 273.15, [22.177490234375, 22.40478515625, 22.7294921875, 22.956787109375, 23.31396484375])
alphas = [linear_interpolation(250, 15.9, 300, 22.5, x) for x in T_air]

print(alphas)