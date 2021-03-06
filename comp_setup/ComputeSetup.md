## Intro

This file documents both how to set up the docker container to run the correct compute environment and also how that container was created. 

The container is called `kmartcontainers/207final`. It's freely available on Dockerhub ([link](https://hub.docker.com/repository/docker/kmartcontainers/207final)). The container is a modified version of the `jupyter/tensorflow-notebook` container created by the jupyter development team

## Using the container

This section gives instructions on how to pull the custom container with the compute environment so that you can run the notebooks that we created.  

### Working on local machine

**Step 0 (first time only):** Install docker on your machine. You can google how to do that.

**Step 1:** First pull the image since it's a big one.

``` shell
docker pull kmartcontainers/207final
```

**Step 2:** Run the container image (with some options)

* `-d` for detatched mode
* `--rm` so the container removes itself when stopped
* `-v` to mount the container to our repo
  * *May need to rename left-hand/local file path.*
* `--name` gives it 'friendly' name to refer to it by
* `-p` forwards the container's port 8889 to our machine's port 8889

``` shell
docker run -d --rm -v ~/w207/mids-w207-final-project:/home/jovyan/work --name vorpal -p 8889:8889 kmartcontainers/207final:latest
```

***Step 2b (Optional):*** Verify that we have mounted correctly. 

Enter the following in the command line to enter the container interactively from the command line.

``` shell
docker exec -it vorpal bash
```

When we navigate down the file structure, we should see our repo files under the `work` subdirectory.

Press `Ctrl` + `d` to exit out of the container and go back to the main terminal screen.

**Step 3:** Start up Jupyter

``` shell
docker exec vorpal jupyter notebook --port=8889
```

**Step 4:** Log in to your jupyter instance.

Go to the website: `localhost:8889` in your computer's browser. 

You'll see the password that you need to login are in the terminal. They will have dumped out under the command that you entered in step 3.

After logging in you should just see a "work" directory. Going into that directory, you should see all of your files out in your mounted directory.

### Working on GCP AI notebooks instance.

**Step 0.1 (First time only):** Install the google Cloud SDK on your local machine.

Instructions can be found here: https://cloud.google.com/sdk/docs

If you're using conda to manage your personal compute environment, you can run `conda install -c conda-forge google-cloud-sdk` to install it.

**Step 0.2 (First time only):** Login to your google account.

Run the following command to bring up a google log in screen and authorize your account.

``` shell
gcloud auth login
```

**Step 0.3 (First time only):** Set up the SSH

The first time that you run the command in "Step 2" below, google will ask to set up an ssh for you. Hit **y** on the keyboard to allow this. 

You can come up with a passphrase to access your ssh key. (*I honestly have no idea if it's necessary, but it felt bad not to do it, so I did set one.*)

I think that you get asked to authorize google to access your project one more time. 

**Step 1:** Start your container on GCP.

From a terminal window in your GCP notebooks instance **perform steps 1 through 3 from the "Working on a local machine"** section of this document.

**Step 2:** SSH into your GCP container.

``` shell
export PROJECT_ID="my-project-id"
export ZONE="my-zone"
export INSTANCE_NAME="my-instance"
gcloud compute ssh --project $PROJECT_ID --zone $ZONE \
  $INSTANCE_NAME -- -L 8889:localhost:8889
``` 

> *Where "my-_____" should be replaced with your specific information*

If you were Kevin for example, you would put:

``` shell
export PROJECT_ID="keen-metric-276522"
export ZONE="us-west1-a"
export INSTANCE_NAME="tensorflow-20200507-151106"
gcloud compute ssh --project $PROJECT_ID --zone $ZONE \
  $INSTANCE_NAME -- -L 8889:localhost:8889
```

*If you want to go to your jupyterlab environment that you normally access via the notebook interface, you can change both of the `8889`s to `8080`s*

*Note, all of this is slightly modified version of instructions from this google website (https://cloud.google.com/ai-platform/notebooks/docs/ssh-access)*

**Step 3:** Go to your notebook.

Go to `localhost:8889` on your browser. (*do same as step 4 in the "Working on a local machine" instructions*)

#### GCP Alternate Access Option: Google Cloud Shell (BEWARE THE TIME-OUT)

If you don't want to download the Google Cloud SDK, you *can* access your notebook via Google Cloud Shell. However, there's a **big drawback** to this method discussed below.

If you want to try this way, you would do the **same as above for "Working on GCP AI notebooks instance"**, except alter the following steps

**Step 0.1 (First time only):** Authorize Google Cloud Shell on your account. 

Instead of installing the Google Cloud SDK...

Go to https://console.cloud.google.com/?cloudshell=true . 

It will open up the google cloud shell and ask you to authorize it on your account.

**Step 3:** 

Instead of going to `localhost` in your web browser...

Click the "Web Preview" button in the upper right corner of the shell window. 

Go down to "Change Port" on the menu that comes up.

Input port 8889 and continue.


The **BIG disadvantage** to this alternate access method is that **the shell times out super quickly** and you need to go in and reconnect frequently. When the shell instance times out, the window that you're viewing the notebook in will time out as well.

## Container Creation

The computing environment is based on the tensorflow notebook put together by the jupyter development team (`jupyter/tensorflow-notebook`). We discovered that we wanted a few extra packages with it, so we extended the container. 

Once we put together the `Dockerfile` (in this directory), we navigate into the directory with the cli and then run the following.

``` bash
docker build . -t 207final
```

``` bash
docker login
```

``` bash
docker tag 207final kmartcontainers/207final:latest
```

``` bash
docker push kmartcontainers/207final:latest
```

## Acknowledgements

* This blog post was a great guide on setting up the Dockerfile to extend an existing image
  * https://kyso.io/KyleOS/running-kysos-docker-image
* The jupyter development team had some insights on adding xgboost
  * https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#xgboost
  * needed to add `conda install -c conda-forge gcc` instead of the original gcc conda install command
* Here's a synopsis of all of the container images that the jupyter team has put together
  * https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html