# Intro to ZHS & Bash


## Agenda
1. What is a shell, accessing it, ZSH and Bash, oh-my-zsh, advantages of CLI interface
2. Prompt: the symbols indicating the Shell you are in
3. man and --help
4. pwd, cd, cd .., cd ~, cd -, absolute and relative paths
5. ls, ls -a, ls -lhtr, notice that directory size is not shown but requires du (disk usage)
6. du -sh *, df -h 
7. ls *.png, notice how * works in general by creating a python program that shows the arguments
8. mkdir, mkdir -p
9. touch
10. cp, cp -r, mv, careful that mv overwrites
11.      rm, rm -f
12. wget https://gist.githubusercontent.com/jacopofar/804c5694ac12a9d6fde653b5a6e3b983/raw/8ffd027bf5e9b1184695e1e55798f699e6acda74/countries_capitals.tsv
13. cat 
14. find . -name "states.txt"
15. vim, vi, and nano: how to save and exit in vim
16. less, head, tail
17. wc
18. The UNIX philosophy: small tools that do one thing and one thing only
19. what is stdin, stdout, stderr
20. combining commands with the pipe
21. grep, egrep
22. echo
23. quiz: what if I want to count how many png files are there in a folder?
24. using > and >> to redirect the output
25. environment variables: set, unset, env, export. Notice that different windows have different variables
26. know the shell with echo $SHELL
27. bashrc, bash_profile, bash_history (and also zsh)
28. the PATH environment variable, the type command
29. aliases
30. SSH: how to connect
31. ssh-keygen
32. the -i flag, the -A flag
33. open a tunnel with -L, sock proxy with -D
34. scp and rsync
35. Creating a script: shebang in bash and python, give it executable permissions
36. https://forms.gle/2AKXXaKj7BSHVoc37
37.

## Terminal plugins
* install "oh my zsh" - https://ohmyz.sh/
    * theme: open `~/.zshrc` and replace `ZSH_THEME="sammy"`
    * you might need to update paths in `.zshrc` from `.bash_profile`
* use `bash` to change back to bash shell
* use `%` to change to ZHS

# Possible problems
* conda not activating:
    * `conda info | grep -i 'base environment'`
    * and run `source {path to anaconda3}` 
    * create Kernels: activate conda env and run `python -m ipykernel install --user --name YourEnvName`

* Check expected paths with `echo $PATH`

## Move and see whats in file system
* `ctrl + d` - exit shell (e.g. if running python go back to system shell)
* `ctrl * r` search through history
* `man ping` - manual for ping function (press q to quit)
* `/a/b/c` absolute path
* `a/b/c` relative path
* `cd -` - bring me to previous folder
* `cd ~/Documents`  or cd will also bring you to home
* `ls -tlr` - shorts list of info (l) by time (t) reversed (r)
* `ls *.csv*`- filters files containign .csv
* `du -sh` - sum (s) of folder size (du) in human readable (h)
* `du -sh *` - as above but splits by element

## filter data --GOOGLE

* `ls *.cs*v*` - filter by name contains .csv
* `ls *folder*` - shows the content inside folder

## Create & Delete
* `/tmp` - folder that is deleted after restart
* `touch hello.txt` - create file OR update edit date
* `rm hello.txt` - delete completely
* `echo "Hello"> sample.txt`- create a quick file with "hello"
* `truncate hello.txt` - update content of a file
* `rmdir` - remove directory
    * `rm -r` - remove a folder and all its content recursively (r)
* `mv file.txt newname.txt` - to move file or rename
* `cp file.txt newfile.txt` - to copy file

## Load and explore files
* `wget` - downlaod file (download from brew)
* `curl URL > name.tsv` - download file from URL
* `cat file.tsv` - print (cat) elements in file
* `less file.tsv` - open content of file
    * `/Fr` - search for Fr in file
* `head file.tsv` - first 10 elements
    * `head -5 file.tsv` - first 5 rows
* `tail file.tsv` - last 10 lines
* `wc file.tsv` - word count in file (rows words letters)
    `wc -l` - just count lines

## Edit files
* `vim file.tsv` - open file -- Q. HOW TO DISPLAY ROW NUMBERS??
    *  `i` - insert mode to edit file
    * `esc` to leave edit mode
    * `:q` - to quite
    * `:!` - to not save changes (`:` to insert flags)
    * `:wq` - write and leave

## joining operators
* `|`- is the pipe operator
* `ls|wc` - get list of elements and count lines
* `ls *.csv|wc -l > number_od_csvs.txt` - get list of csv files in folder, pipe, get just the lines, save it as file.txt
    * `>` - create if does not exist
    * `>>` - append content to new file
* `wc file.tsv|less`

### filtering
* `grep Lon file.tsv` - to filter elements containing "Lon"
    * `grep -i` - for avoiding caps
    * `grep -v` - negation
    * `grep -iv ireland file.tsv` - select all not containing "ireland" up or lower case 
    * `grep -iv ireland|grep Mon` - you can concatenate multiple
    * `cat countries_capitals.tsv|grep Ireland > only_ireland.tsv` filter all elements that contain "Ireland" and save it to a new file
* `cat file.tsv|sort` - print content sorted
    `|uniq` - shows unique elements
* `cat .txt | head -n 1 | jq` show structure of json file


## Envs and Vars
* `echo` - the print funciton for bash
* `echo $HOME`
* `echo $SHELL`
* `echo $PATH`  
* `export` - to set session variable for **current session** only
    * you can add it to the .zshrc file to make it always available
* `alias greet='hello $USER'` - create a shortcut 'greet'
    * you can also set them permanently in the .zshrc or bash profile

## Execute files
* `python3 file.py` - use shell to run file
* `chmode +x file.py` - force file to execute directly
* `file file.py` - system detects type of file
    * `#!/bin/env python3` - add to file for system to recognise

## Connect to remote machine
* `whoami` - print machine name
* `ssh` - to connect external server
* `top` - commands taking most resources currently
* `sc` - secure copy from ssh
    * Example: `scp re.mote.server:/file/path.txt local/path/file.txt` 
* `rsync` - make a copy only of new changes (for backup)

### Create Key
* `ssh-keygen` - create key 
* `.ssh` - here are keys located, home/.ssh
* `cat id_rsa.pub` - to print & copy/paste in GitHub
* `ssh git@github.com` - test key in terminal

