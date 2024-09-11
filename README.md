
## Cluster Running
SSH into GPU Cluster, and run project on GPU cluster: 

Reference: <a> https://www.imperial.ac.uk/computing/people/csg/guides/hpcomputing/gpucluster/ </a>
```sh
ssh <username>@gpucluster2.doc.ic.ac.uk
cd <directory>             # output of cluster will be redirected here at the current directory
sbatch clusterRunner.sh
squeue                     # To view and ensure current working jobs
```
In case of changes to `main.py` requiring parameters in command line, configure `clusterRunner.sh` and convert to executable
```sh
chmod +x clusterRunner.sh
```
<b>OPTIONALLY</b> can also use `salloc` from the `gpucluster2.doc.ic.ac.uk` shell. However, make sure to note down `NODELIST(REASON)` in this case to directly navigate to 
`<NODELIST(REASON)@doc.ic.ac.uk` when returning while the job is still running.
***
## Background Downloader
`downloader.py` optionally available for running on DOC machines in the background as long running task for downloading dataset. `RUNME` has already been configured 
and need to simply run it.
```sh
chmod +x RUNME            # in case of changes to RUNME, make executable again.
nice nohup ./RUNME </dev/null >&/dev/null &
```
Configure dataset classes and path from `donwloader.py` to download specific datasets from OMNIART.

