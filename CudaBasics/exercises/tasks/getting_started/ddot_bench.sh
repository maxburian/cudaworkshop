for n in 1000 10000 100000 1000000 10000000 100000000 1000000000; do ./ddot $n|grep took; done | tee ddot_bench.dat
python3 ddot_bench.py &
