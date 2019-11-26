#PBS -m e
#PBS -P y57
#PBS -q express 
#PBS -l walltime=00:10:00
#PBS -l ncpus=16
#PBS -l mem=16GB
#PBS -l wd
#PBS -N job
#PBS -l jobfs=1GB
#PBS -l other=hyperthread

./batch_dask.sh &> batch_dask.log
