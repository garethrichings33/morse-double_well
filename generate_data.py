'''
Generate a CSV file of values of a 2D Morse curve and double well.
Points are chosen randomly.
'''


def generate_data(filename):
    import csv
    import math
    import random

# Set random seed for repeatability when testing.
    random.seed(0)

    def surface(x, y):
        morse = 0.5 * (1. - math.exp(-0.5*(x-2)))**2
        double_well = 0.012 * y**4 - 0.2 * y**2 + 0.1*y
        return morse + double_well

    num_points = 1000
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "value"])

        for i in range(num_points):
            x = random.uniform(-0.5, 20.)
            y = random.uniform(-5., 5.)
            value = surface(x, y)
            writer.writerow([x, y, value])

    return filename


if __name__ == '__main__':
    generate_data('surface.csv')
