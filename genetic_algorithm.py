from rbfn import RBFN
from toolkit import toolkit as tk
import numpy as np
import random
from drawplot import drawPlot

dataset, answers = tk.readFile("./train4dAll.txt")
K=10
reproduction_rate = 0.1

"""
    產生隨機初始族群:
    w_list為隨機產生
    m_list為選擇隨機train dataset之點
    std_list為隨機在5~20間產生
    n: 族群數量
    K: 群聚數量
"""
def initial_group(n = 50):
    result = []
    for i in range(n):
        w_list = np.random.randn(K+1)
        randomIntArray = [random.randint(0,len(dataset)-1) for k in range(K)]
        m_list = np.array(dataset[randomIntArray]).flatten()
        std_list = np.array([np.random.uniform(5,20) for i in range(K)])
        genetic = np.concatenate((w_list, m_list, std_list))
        result.append(genetic)
    return result

"""
    複製步驟:
    選擇基因池中前 n % 複製
    top_percent: 要被複製的%數
"""
def reproduce(genetic_pool, top_percent = 0.1, fitness_values=[]):
    if(len(fitness_values) == 0):
        fitness_values = []
        for i in range(len(genetic_pool)):
            fitness_values.append(fitness_function_step(genetic_pool[i]))
        fitness_values = np.array(fitness_values)
    
    # 選出前 n %的元素
    num_top_params = int(len(fitness_values) * top_percent)
    #選最大的前 n %
    top_indices = np.argpartition(fitness_values, -1 * num_top_params)[-1 * num_top_params:]
    #選最小的前 n %
    # top_indices = np.argpartition(fitness_values, num_top_params)[:num_top_params]

    # 根據索引找出對應的參數
    top_params = [genetic_pool[i] for i in top_indices]
    
    #複製前 n %的基因
    result = np.tile(top_params, int(1/top_percent)).reshape(-1, 5*K+1)
    
    print("reproduce")
    return result

"""
    交配步驟:
    選擇基因池中50%的基因會被隨機兩兩配對進行交配
    crossover_rate: 交配率
"""
def crossover(genetic_pool, crossover_rate = 0.5):
    num_crossover_params = int(len(genetic_pool) * crossover_rate)

    if num_crossover_params % 2 != 0:
        num_crossover_params += 1

    crossover_pool = np.array(genetic_pool[:num_crossover_params])
    
    #將crossover_pool隨機排列後兩兩一組
    crossover_pool = crossover_pool[np.random.permutation(crossover_pool.shape[0])].reshape(-1,2,5*K+1)

    #進行交配
    for i in range(len(crossover_pool)):
        x1 = crossover_pool[i][0]
        x2 = crossover_pool[i][1]
        randvar = random.random()
        if randvar > 0.5:
            sigma = random.random()
            x1 = x1 + sigma * (x1 - x2)
            x2 = x2 - sigma * (x1 - x2)
        else:
            sigma = random.random()
            x1 = x1 + sigma * (x2 - x1)
            x2 = x2 - sigma * (x2 - x1)
        crossover_pool[i][0] = x1
        crossover_pool[i][1] = x2
        break
    crossover_pool = crossover_pool.reshape(-1,5*K+1)
    genetic_pool[:num_crossover_params] = crossover_pool
    print("crossover")
    return genetic_pool
    
"""
    突變步驟:
    選擇基因池中 n %的基因會被隨機抽取進行突變
    mutation_rate: 突變率
"""
def mutation(genetic_pool, mutation_rate = 0.1):
    num_mutation_genetic = int(len(genetic_pool) * mutation_rate)
    genetic_pool = np.array(genetic_pool)
    randomIntArray = [random.randint(0,len(genetic_pool)-1) for i in range(num_mutation_genetic)]
    mutation_pool = genetic_pool[randomIntArray]
    for i in range(len(mutation_pool)):
        rand_noise = np.random.rand(5*K+1) * 10 - 5
        mutation_pool[i] = mutation_pool[i] + rand_noise
    genetic_pool[randomIntArray] = mutation_pool
    print("mutation")
    return genetic_pool.tolist()

"""
    將基因解碼成RBFN之元素
"""
def decode(genetic):
    genetic = np.array(genetic)
    delta = genetic[0]
    w_list = genetic[1:(1+K)]
    m_list = genetic[K+1:3*K+(K+1)].reshape(-1,3)
    std_list = genetic[3*K+(K+1):]
    return delta, w_list, m_list, std_list


#讓基因產生predict與dataset中的ans 計算 mean loss
def fitness_function(genetic):
    delta, w, m, std = decode(genetic)
    ME = 0
    i = 0
    for data, ans in zip(dataset, answers):
        F = RBFN.forward(data, m, std, w, delta)
        E = (ans - F)**2/2
        ME = ME + E
        i+=1
    ME = ME / len(dataset)
    return ME

#讓基因實際去跑map
def fitness_function_step(genetic, mapFilePath="map.txt"):
    delta, w, m, std = decode(genetic)
    rbfn = RBFN()
    rbfn.setParams(delta, w, m, std)
    original_point, goal_points, boarder_points = drawPlot.readMapFile(mapFilePath)
    currentPoint = original_point[:-1]
    currentPhi = original_point[-1]
    original_point = original_point[:-1]
    currentVector = np.array([100, 0])
    step = 0
    bonus = 0
    while True:
        try:
            phi = currentPhi
            point = currentPoint
            sensor_vectors = drawPlot.getSensorVector(currentVector, phi)
            sensor_distances = []
            cross_points = []
            for sensor_vector in sensor_vectors:
                cross_point = drawPlot.findCrossPoint(boarder_points, point, sensor_vector)
                distance = tk.euclid_distance(cross_point, point)
                cross_points.append(cross_point)
                sensor_distances.append(distance)
            sensor_distances = np.array(sensor_distances).flatten()
            theta = rbfn.predict(sensor_distances)
            if drawPlot.is_Crash(point, boarder_points) == True:
                raise Exception("touch the wall of map")
            currentPoint, currentPhi = drawPlot.findNextState(point, phi, theta)
            if currentPoint[1] > min(goal_points[:,1]):
                bonus = 1000000
                break
            step +=1
        except Exception as e:
            # print(e)
            break
    result = 0.5 * step + 0.5 * float(tk.euclid_distance(currentPoint, original_point)) + bonus
    if bonus != 0:
        result -= step * 10
    return result

if __name__ == "__main__":
    genetic_pool = initial_group(50)
    max_epochs = 100
    fitness_values = []
    for i in range(max_epochs):
        genetic_pool = reproduce(genetic_pool, fitness_values)
        genetic_pool = crossover(genetic_pool)
        genetic_pool = mutation(genetic_pool)
        fitness_values = np.array([fitness_function(genetic) for genetic in genetic_pool])
        best_genetic = genetic_pool[np.argmin(fitness_values)]
        print(f'{i}/{max_epochs-1}: Loss = {np.min(fitness_values)}')
        print('--------')

    
        