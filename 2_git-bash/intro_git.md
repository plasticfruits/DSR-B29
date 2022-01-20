# Intro to git

## Basic commands
* `git init` - start version control
* `git clone` - to copy a repository on your machine
* `git status` - show the current state of the repository
* `git add` - add a file or its changes to the staged commit
    * `git add .` - stage everything
* `git commit` - snapshots the current state
    * `git commit -a` - commit everything
* `git fetch` - retrieve the metadata from the remote repository
* `git push/git pull` - sync to/from your local repository
* `git checkout` - change the local branch
    * `git checkout -b new-branch` - create if branch does not exist
    * `git checkout master`- switch to master branch
* `git merge` - merge with another branch
    * `git merge branch-name`
* `git branch` - create, list or delete branches
    * `git branch -m new-namw`- to rename current branch
* `git log` - show history of repo

## Check visual status of repo
* `git log --all --oneline --graph` - visual preview of state


## Clone methods examples
* SSH is preferred over HTTPS


## Create SSH key
* `ssh-keygen` - create key 
* `.ssh` - here are keys located, home/.ssh
* `cat id_rsa.pub` - to print & copy/paste in GitHub
* `ssh git@github.com` - test key in terminal
