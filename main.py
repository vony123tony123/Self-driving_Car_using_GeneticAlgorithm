# -*- coding: utf-8 -*-
import time
import traceback
import threading
from tqdm import tqdm

import matplotlib
from PyQt5.QtCore import pyqtSignal, QThread, QTimer, QEventLoop

import numpy as np
from Windows import Ui_MainWindow
from rbfn import RBFN
import genetic_algorithm as GA
from drawplot import drawPlot
import os
from toolkit import toolkit
# 导入程序运行必须模块
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
matplotlib.use('Qt5Agg')

# 設定gui的功能
class MyMainWindow(QMainWindow, Ui_MainWindow):

    step=0#用來判斷要不要創建plotpicture

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.inputfilepath_btn_2.clicked.connect(self.choosefileDialog)
        self.pushButton_2.clicked.connect(self.startCar)
        self.canva = drawPlot()
        self.plot_layout.addWidget(self.canva)

    def readMapFile(mapFile):
        goal_points = list()
        boarder_points = list()
        with open(mapFile, 'r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = list(map(float, line.strip('\n').split(',')))
                if i == 0 :
                    original_point = line
                elif i == 1 or i == 2 :
                    goal_points.append(line)
                else:
                    boarder_points.append(line)
        original_point = np.array(original_point)
        goal_points = np.array(goal_points)
        boarder_points = np.array(boarder_points)
        return original_point, goal_points, boarder_points

    def choosefileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if filename:
            self.mapfilepath_edit.setText(filename)
        else:
            self.mapfilepath_edit.setText("")

    def startCar(self):
        mapFilePath = self.mapfilepath_edit.text()
        original_point, self.goal_points, self.boarder_points = drawPlot.readMapFile(mapFilePath)
        self.canva.drawMap(self.goal_points,self.boarder_points)

        genetic_pool = GA.initial_group(1000)
        max_epochs = 100
        fitness_values = []
        previous_best_fitness_value = 0
        mutation_rate = 0.3
        count_goal_epochs = 0
        for i in range(max_epochs):
            print("------")
            genetic_pool = GA.reproduce(genetic_pool, fitness_values=fitness_values, top_percent = 0.25)
            genetic_pool = GA.crossover(genetic_pool)
            genetic_pool = GA.mutation(genetic_pool, mutation_rate = mutation_rate)
            print(f'mutation_rate = {mutation_rate}')

            progress = tqdm(total=len(genetic_pool))

            for genetic in genetic_pool:
                value = GA.fitness_function_step(genetic)
                fitness_values.append(value)
                progress.set_description('Fitness')
                progress.update(1)
            fitness_values = np.array(fitness_values)
            best_genetic = genetic_pool[np.argmax(fitness_values)]
            print("")
            print(f'{i}/{max_epochs-1}: steps = {np.max(fitness_values)}')

            if max(fitness_values) > 10000:
                with open('goal_result.txt', 'a') as f:
                    f.write(f'{best_genetic}\n')
                count_goal_epochs += 1            

            if max(fitness_values) <= previous_best_fitness_value:
                mutation_rate += 0.05
                if mutation_rate > 0.9:
                    mutation_rate = 0.9

            previous_best_fitness_value = max(fitness_values) if len(fitness_values) != 0 else 0
            fitness_values = []
            if count_goal_epochs >= 5:
                break
        
        self.RBFN = RBFN()
        delta, w_list, m_list, std_list = GA.decode(best_genetic)
        self.RBFN.setParams(delta, w_list, m_list, std_list)

        self.currentPoint = original_point[:-1]
        self.currentPhi = original_point[-1]
        self.currentVector = np.array([100, 0])
        self.canva.clearPoints()

        self.loop = QEventLoop()
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start()
        self.loop.exec_()
   
    def updatePlot(self):
        try:
            phi = self.currentPhi
            point = self.currentPoint
            self.canva.drawPoint(point)
            sensor_vectors = drawPlot.getSensorVector(self.currentVector, phi)
            sensor_distances = []
            cross_points = []
            for sensor_vector in sensor_vectors:
                cross_point = drawPlot.findCrossPoint(self.boarder_points, point, sensor_vector)
                #self.canva.drawPoint(cross_point, 'r')
                distance = toolkit.euclid_distance(cross_point, point)
                cross_points.append(cross_point)
                sensor_distances.append(distance)
            sensor_distances = np.array(sensor_distances).flatten()
            theta = self.RBFN.predict(sensor_distances)
            if drawPlot.is_Crash(point, self.boarder_points):
                raise Exception("touch the wall of map")
            self.canva.updatePlot()
            self.currentPoint, self.currentPhi = drawPlot.findNextState(point, phi, theta)
            if self.currentPoint[1] > min(self.goal_points[:,1]):
                self.timer.stop()
                self.loop.quit()
        except Exception as e:
            print(e)
            traceback_output = traceback.format_exc()
            #print(traceback_output)
            self.timer.stop()
            self.loop.quit()
            # sys.exit(app.exec_())

if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainWindow()

    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())

