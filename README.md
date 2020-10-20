# Connecting to IDUN
(Must be on NTNU network or use VPN!)
https://www.hpc.ntnu.no/idun

Log on:

`ssh -l USERNAME idun-login1.hpc.ntnu.no`

The directory where you're allowed to do computations:

`cd /lustre1/work/USERNAME`

Here, clone into this github repo, by using 

`git clone https://github.com/maudrod/prosjektoppgave.git`

Connecting to IDUN when you are at home and not at NTNU:

1) `ssh maudr@markov.math.ntnu.no`
2) `ssh -l USERNAME idun-login1.hpc.ntnu.no`


## Creating a virtual environment in IDUN
https://www.hpc.ntnu.no/idun/getting-started-on-idun/modules
`$ module load intel/2018b
  $ module load Python/3.6.6
  $ module list
  Currently Loaded Modules:
    1) GCCcore/.7.3.0                    5) impi/2018.3.222   9) zlib/.1.2.11      13) GMP/.6.1.2
    2) binutils/.2.30                    6) imkl/2018.3.222  10) ncurses/.6.1      14) libffi/.3.2.1
    3) icc/2018.3.222-GCC-7.3.0-2.30     7) intel/2018b      11) libreadline/.7.0  15) Python/3.6.6
    4) ifort/2018.3.222-GCC-7.3.0-2.30   8) bzip2/.1.0.6     12) XZ/.5.2.4
  $ type virtualenv
  virtualenv is /share/apps/software/MPI/intel/2018.3.222-GCC-7.3.0-2.30/impi/2018.3.222/Python/3.6.6/bin/virtualenv
  $ virtualenv datasci
  Using base prefix '/share/apps/software/MPI/intel/2018.3.222-GCC-7.3.0-2.30/impi/2018.3.222/Python/3.6.6'
  New python executable in /lustre1/work/bjornlin/pythonenvs/datasci/bin/python
  Installing setuptools, pip, wheel...done.
  $ ls
   datasci
  $ source datasci/bin/activate
  (datasci)$ pip install scipy numpy scikit-learn pandas matplotlib
  (datasci)$ deactivate`


## Creating a job in IDUN
https://www.hpc.ntnu.no/idun/getting-started-on-idun/running-jobs

Create a job.slurm file:

`vim job.slurm`
Use the key i to edit this file. Copy paste the file as it is on the help page. 
Account = ie-imf
Email: your ntnu email

When done, Ctrl+C, then write `:wq!` and hit enter.

`chmod u+x job.slurm`
`sbatch job.slurm`


# Making changes to the repository

First make sure you're up to date with the current repository:

`git pull`

Once you've made a change, use the command

`git add .`

(to add everything you've done) or 

`git add FILEYOUMADECHANGESTO.py`

`git commit -m "la til det her"`

`git push`



# Saving outputs from file runs

`python3 maud.py output > maudteksten.txt`

