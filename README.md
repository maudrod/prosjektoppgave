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

1) `ssh markov`
2) `ssh -l USERNAME idun-login1.hpc.ntnu.no`


## Creating a virtual environment in IDUN



## Creating a job in IDUN



# Making changes to the repository

Once you've made a change, use the command

`git add .`

(to add everything you've done) or 

`git add FILEYOUMADECHANGESTO.py`

`git commit`

`git push`

