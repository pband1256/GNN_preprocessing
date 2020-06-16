log = job_logs/log/GNN_$(Cluster)_$(Process).log
output = job_logs/output/GNN_$(Cluster)_$(Process).out
error = job_logs/error/GNN_$(Cluster)_$(Process).error

executable = /home/jovyan/job_scripts/test.py
request_GPUs = 2
request_memory = 100GB

Universe  = vanilla
+NATIVE_OS = true
requirements = (OpSysMajorVer =?= 7)
notification = Error
notify_user = lehieu1@msu.edu

#GNN/gnn_icecube/htcondor_multi_main.sh

queue 3