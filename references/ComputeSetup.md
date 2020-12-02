## Intro

The computing environment is based on the tensorflow notebook put together by the jupyter development team (`jupyter/tensorflow-notebook`). We discovered that we wanted a few extra packages with it, so we extended the container. 

Once we put together the (small) Dockerfile  in this directory, we navigate into the directory with the cli and then run the following.

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