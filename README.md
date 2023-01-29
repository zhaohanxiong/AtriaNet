![image](https://user-images.githubusercontent.com/29684281/211007087-f757fad4-9697-4d98-b3f8-8dd3a415fccb.png)

##### ssh key
```
zxio506@bioeng20.bioeng.auckland.ac.nz
```

##### ssh into titan V computer
```
ssh -X zxio506@BN356574.uoa.auckland.ac.nz
```

##### activate virtual environment
```
source /home/zxio506/Virtual_ENV/TitanV/bin/activate
```

##### location where scripts are stored
```
cd /hpc/zxio506/2022_runs/
```

##### run model
```
bash run.sh
```

##### file transfer
```
scp -r zxio506@bioeng20.bioeng.auckland.ac.nz:/hpc/zxio506/2022_runs/output_here .
```

```
scp -r zxio506@bioeng20.bioeng.auckland.ac.nz:/hpc/zxio506/2022_runs/*.py .
scp -r zxio506@bioeng20.bioeng.auckland.ac.nz:/hpc/zxio506/2022_runs/*.sh .
```
