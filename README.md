# NLP大作业（自实现单、双层LSTM）
学院：计算机科学与工程

班级：人工智能2001

姓名：许子强

学号：20201111


LSTM.py为包含单层、双层LSTM模型的文件，在源码文件夹下也可找到（LSTM\LSTM.py）

在LSTM.py文件中，LSTM_one为单层模型，LSTM_double为双层模型

文件中初始训练为双层模型，如需训练单层模型，只需在LSTM.py文件的train_LSTM中，把第一行model = LSTM_double()改为model = LSTM_one()即可
