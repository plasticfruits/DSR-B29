# **Docker Notes**

Cheatsheet:  
https://dockerlabs.collabnix.com/docker/cheatsheet/

<br>  

### **About**
Runs a linux process in an isolated and  controlled way (container)
A container is a virtual env with the requirements needed to run your application locally, in production or by sharing it.

The requirement state often includes sub-libraries that are not stated (like C++ libs)

<br>

### **Definitions:**
* Daemon: a server that is **always** running (aka engine)
* Client: primary way to interact with Daemon (i.e. Docker client)
    * communicates with engine via HTTP
* Image: contains your application, libraries and interpreters (i.e. everything you need to run the application)
    * Docker app: www.hub.docker.com
    * `{docker hub username}/{image}:{tag}` - if not tag docker will fetch the latest.
    * `docker pull python:3.5` - no username as its an official image
* Container: a running image

<br><br>  

## **A. Using an image**
<br>  

### **Commands:**
* `docker pull` to download an image from the Docker Hub
    * Example: `docker pull python:3.5`
* `docker images` list downloaded images
* `docker rmi` to delete an image
    * Example: `docker rmi python:3.5`
* `docker run` start a container  
    * `-it` runs in interactive mode
    * `--rm` delete upon exit
    * `--name` to give it a name
    * Good to combine all together: `docker run -it --rm --name mypython35 python:3.5`
        * you can run another shell inside the container: 
        `docker run -it --rm --name mypython35 python:3.9 bash`
    * `-e MY_VAR=42` to define environment variables when starting a container with `docker run`
    * `-v /Documents/temp:/documents` to mount folders / volumes in container
    * `-p 9000:8000` to expose a specific IP address which you can access via http://127.0.0.1:9000/
    * `--link` to connect two containers 
            - `docker run -it --rm --link another-container python:3.9 bash`
            - you can start server with `python -m http.server` in guest and use curl http://another-container:8000 to access it via linked container
* `docker ps` to see running containers
    * `docker ps -a` see all containers (running and not running)
* `docker stop` ask program to terminate
* `docker kill` - terminate program abruptly/immediately    
* `docker inspect` to see all details of active container (very useful!)
* `docker rm $(docker ps -aq)` can delete all stopped containers to save space
* `docker exec` to access/attach to an existing container
    *  `docker exec -it container_name bash`
* `docker logs` to see output of container (for trouble-shooting)

<br>  

### Bash (inside docker)
* `docker exec -it container_name bash` connect to a db with bash
* `ps -ax` to see everything running in container
* `mysql -uexample-user -p` to connect with mysql client

* `python3 -m http.server` to create server with files
* `python3 -m venv .venv` to create a python3 virtual env in folder
* `source .venv/bin/activate` to activate virtual env created


<br>  

### Mount Folders / Volumes
You can "mount" local folders and volumes to a docker container when starting
* `docker run -it --rm --name py39 -v /Users/alcachofa/Documents/temp:/documents python:3.9 bash` this will mount all files from local folder to a container directory -- changes will by bijective! 
    * you can use multiple paths tags to create multiple mounts in same command
* `docker run -it --rm --name py39 -v volumename:/documents python:3.9 bash` mounts a volume instead of a folder

<br>  

### MariaDB
Always requires `;` after command  
Whenever we stop a container with a DB we lost the DB
* `show databases test-db;`
* `use information_schema;`
* `create table bla(age integer);` to create a table

<br>  

### Docker Compose
For dealing with multiple containers

<br>  

### Kubernetes
check out minikube for testing locally

<br><br>  

## **B. Building an image**
Usually done by defining a Dockerfile

* Structure of a Docker file:
    ```
    FROM python:3.9 {an existing image}
    RUN ... {install libs}
    COPY ... {copy files using relative path}
    RUN ... {define executable}
    CMD ... {default command to execute}
    ``` 
* Example:
    ```
    FROM python:3.9
    COPY hello_world.py /app/hello_world.py
    CMD ["python3", "/app/hello_world.py"]
    ```
* `vim Dockerfile` (no extension needed) adds a dockerfile in root directory of project (to use relative paths)
* `.dockerignore` use to ignore files
* `docker build .` to build the iage in current directory
    * `docker build -t --rm imagename .` to give name to image
* `docker tag` to give name to an already-existing image
* `docker run -it imagename` to run image and close 
* `docker run -it imagename bash` to access with bash
* `docker push imagename:1.2` to push to docker repo

<br><br>  

## Flask webapp
* `FLASK_APP=myapp:app flask run --host 0.0.0.0` run with flask an app 'app' inside 'myapp' on any port
* run flask app
    * `docker run -it --rm myflaskapp bash` to check directory
    * `docker run -it -p 5000:5000 --rm myflaskapp` run on specific port
* Example:
    ```
    FROM python:3.9
    COPY . /myapp #copy all to /myapp
    RUN pip install -r /myapp/requirements.txt
    ENV FLASK_APP=myapp:app 
    WORKDIR /myapp #default dir
    CMD ["flask","run", "--host", "0.0.0.0"]
    EXPOSE 5000 #port
    ```

