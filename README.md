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

`$ type virtualenv`

`$ virtualenv datasci`

`$ source datasci/bin/activate`

`(datasci)$ pip install scipy numpy scikit-learn pandas matplotlib`

`source datasci/bin/activate`


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

# Move files 
Write in terminal: `mv <filename> <map>



