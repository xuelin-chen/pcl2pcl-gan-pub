import os

num_workers = 70

for i in range(0,num_workers):
    #a = 'nohup python3 virtual_scan_shapenet_objects.py --worker_id %d --num_workers %d > log%d.txt 2>&1 &' % (i, num_workers, i)
    a = 'nohup python3 virtual_scan_shapenet-v1_objects.py --worker_id %d --num_workers %d > log%d.txt 2>&1 &' % (i, num_workers, i)
    os.system(a)