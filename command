nohup python3 -u main_synthetic.py 500 500 3 5 0 0.3 >> result/N500T500k3lam[0,0.3].txt 2>&1 &
nohup python3 -u main_synthetic.py 500 500 3 5 0.3 0.6 >> result/N500T500k3lam[0.3,0.6].txt 2>&1 &
nohup python3 -u main_synthetic.py 500 500 3 5 0.6 0.9 >> result/N500T500k3lam[0.6,0.9].txt 2>&1 &

nohup python3 -u main_synthetic.py 200 1250 3 5 0 0.3 >> result/N200T1250k3lam[0,0.3].txt 2>&1 &
nohup python3 -u main_synthetic.py 200 1250 3 5 0.3 0.6 >> result/N200T1250k3lam[0.3,0.6].txt 2>&1 &
nohup python3 -u main_synthetic.py 200 1250 3 5 0.6 0.9 >> result/N200T1250k3lam[0.6,0.9].txt 2>&1 &

python main_realworld.py 1 3 0 0.9 1000
python main_realworld.py 1 3 0 0.9 500
python main_realworld.py 2 3 0 0.9 1000
python main_realworld.py 2 3 0 0.9 500