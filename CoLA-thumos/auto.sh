#!/bin/bash
# 以下文件是串行执行的。 nohup sh auto.sh &
python3 main_cola.py test  0
python3 main_cola.py train 1
python3 main_cola.py test  1
python3 main_cola.py train 2
python3 main_cola.py test  2
python3 main_cola.py train 3
python3 main_cola.py test  3
python3 main_cola.py train 4
python3 main_cola.py test  4
python3 main_cola.py train 5