import numpy as np
import math
import matplotlib.pyplot as plt
import time

# 初始种群数
init_population = 200

# 变异概率
mutation_pro = 0.05

# 迭代次数
iterate_num = 20000


# 解决TSP问题的遗传算法类
class GeneticAlgTSP:
    def __init__(self, filename):
        # 读取文件数据
        with open(filename, 'r') as file:
            lines = file.readlines()
            coordinate_index = 0
            i = 0
            for line in lines:
                if line == 'EOF':
                    break
                if line.startswith("DIMENSION"):
                    self.dimension = int(line.split(":")[1])  # 城市数量n
                    self.city_coordinates = np.zeros((self.dimension, 2))  # 城市坐标，n×2矩阵，分别是横坐标和纵坐标
                    self.city_distances = np.zeros((self.dimension, self.dimension))  # 城市之间的距离，n×n矩阵
                if coordinate_index == 1:
                    city_features = line.strip().split()
                    self.city_coordinates[i][0] = float(city_features[1])  # 城市横坐标
                    self.city_coordinates[i][1] = float(city_features[2])  # 城市纵坐标
                    i += 1
                if line.startswith("NODE_COORD_SECTION"):
                    coordinate_index = 1
        # 计算得到两两城市之间的距离，存入city_distances二维数组中
        for i in range(self.dimension):
            for j in range(self.dimension):
                x = self.city_coordinates[i][0] - self.city_coordinates[j][0]
                y = self.city_coordinates[i][1] - self.city_coordinates[j][1]
                distance = math.sqrt(x * x + y * y)
                self.city_distances[i][j] = distance
        # 随机初始化种群
        self.population = np.zeros((init_population, self.dimension)).astype(int)
        for i in range(len(self.population)):
            self.population[i] = np.random.choice(range(self.dimension), size=self.dimension, replace=False)
        # 得到初始种群每个个体（路径）的长度，并计算得到当前最佳路径best_route
        self.get_all_distances()
        best_index = np.argmin(self.distances)
        self.best_route = self.population[best_index]  # 目前已知最佳路径

    # 计算某一条路径的长度
    def get_one_distance(self, route):
        length = 0
        for i in range(len(route) - 1):
            length += self.city_distances[route[i]][route[i + 1]]
        length += self.city_distances[route[-1]][route[0]]
        return length

    # 计算种群中每条路径的长度
    def get_all_distances(self):
        self.distances = np.zeros(init_population)  # 种群此时每条路径的长度的值
        for i in range(len(self.population)):
            self.distances[i] = self.get_one_distance(self.population[i])
        return self.distances

    # 计算每条路径的适应度的值（长度的倒数）
    def get_fitness(self):
        fitness = np.zeros(init_population)    # 种群此时每条路径的适应度的值
        for i in range(len(self.distances)):
            fitness[i] = 1 / self.distances[i]
        return fitness

    # 对初始种群进行选择
    def select(self):
        # 定义选择后的种群
        selected_population = np.zeros((len(self.population), self.dimension)).astype(int)
        # 将每条路径的适应度在总适应度之和的占比作为选择的概率
        fitness=self.get_fitness()
        probability = fitness / np.sum(fitness)
        for i in range(init_population):
            # 依照概率选择种群中的一条路线作为新种群的一条路线
            choice = np.random.choice(range(init_population), p=probability)
            selected_population[i] = self.population[choice]
        return selected_population

    # 对某两条路径进行交叉
    def crossover(self, route1, route2):
        # 随机选择一个断点
        point = np.random.randint(1, self.dimension - 1)
        # 深拷贝原来的路径
        child1 = np.copy(route1)
        child2 = np.copy(route2)
        j = k = 0
        # 交叉得到新路径
        for i in range(self.dimension):
            if route2[i] not in child1[:point]:
                child1[point + j] = route2[i]
                j += 1
            if route1[i] not in child2[:point]:
                child2[point + k] = route1[i]
                k += 1
        # 将新路径替换原来的路径
        route1 = child1
        route2 = child2

    # 对种群中所有个体进行两两交叉
    def crossover_all(self):
        for i in range(0, self.population.shape[0] - 1, 2):
            self.crossover(self.population[i], self.population[i + 1])

    # 对种群进行变异（倒置变异）
    def mutation(self):
        # 得到决定是否变异的随机数数组
        pro_array = np.random.rand(init_population)
        for i in range(self.population.shape[0]):
            # 在一定的变异概率下
            if pro_array[i] <= mutation_pro:
                # 随机选取变异片段
                point1 = np.random.randint(0, self.dimension)
                point2 = np.random.randint(point1 + 1, self.dimension + 1)
                old_seq = self.population[i][point1:point2]
                # 将变异片段倒置
                new_seq = old_seq[::-1]
                self.population[i][point1:point2] = new_seq

    def iterate(self, num_iterations, print_best_route=False):
        plt.ion()  # 开启交互模式
        best_distance_over_time = []  # 初始化存储最佳适应度值的列表

        for i in range(num_iterations):
            # 依次进行遗传算法中选择，交叉，变异，得到适应度（路线长度），更新最佳路径的步骤
            self.population = self.select()
            self.crossover_all()
            self.mutation()
            self.get_all_distances()
            best_index = np.argmin(self.distances)
            new_best_route = self.population[best_index]  # 目前已知最佳路径
            distance_old = self.get_one_distance(self.best_route)
            distance_new = self.distances[best_index]
            if distance_old > distance_new:
                self.best_route = new_best_route
            best_distance_over_time.append(self.distances[best_index])  # 存储最佳适应度值的倒数（如果适应度是路线长度）

            if i % 200 == 0:  # 每200次迭代更新一次图形
                print('迭代次数：', "{:5d}".format(i), '  ', end='')
                print('最短路径值：', self.get_one_distance(self.best_route))
                if print_best_route:
                    print(self.best_route)
                self.plot_best_route()  # 调用绘制最佳路径的方法
                plt.pause(0.1)  # 暂停一段时间，以便图形更新
                plt.clf()  # 清除当前图形，准备下一次绘制

        plt.ioff()  # 关闭交互模式
        # 输出最终得到的最佳路径城市列表和最短路径值
        final_distance = self.get_one_distance(self.best_route)
        print(list(self.best_route))
        print(final_distance)
        self.plot_best_route()  # 最后再绘制一次，确保最终结果被显示

        # 绘制适应度曲线
        plt.figure()  # 新建一个图形
        plt.plot(best_distance_over_time, label='Best Distance Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Distance Over Time')
        plt.legend()
        plt.show()  # 显示图形

    # 使用matplotlib绘制最佳路径函数
    def plot_best_route(self):
        plt.figure(figsize=(10, 6))
        # 绘制所有城市的位置
        plt.scatter(self.city_coordinates[:, 0], self.city_coordinates[:, 1], c='red', label='Cities')
        # 绘制最佳路径
        for i in range(-1, len(self.best_route) - 1):
            start_city = self.city_coordinates[self.best_route[i]]
            end_city = self.city_coordinates[self.best_route[i + 1]]
            plt.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], 'k-')
        plt.title('Best Route: ' + str(self.get_one_distance(self.best_route)))
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    start_time = time.time()
    a = GeneticAlgTSP('wi29.tsp')
    a.iterate(iterate_num)
    end_time = time.time()
    print("运行时间：", end_time - start_time, "秒")
