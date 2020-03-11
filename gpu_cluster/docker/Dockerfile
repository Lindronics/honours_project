# FROM tells Docker which base image to use for this build
FROM tensorflow/tensorflow:latest-gpu-py3

# COPY is used to copy tftest.py from the local machine
# to a location inside the image
# COPY tftest.py /tmp

# ENV can be used to set environment variables
# ENV TFTEST_ENV_VAR 12345

# WORKDIR configures the current working directory that 
# the CMD will be executed within. Since the COPY
# command above puts the script under /tmp, we'll set
# it to the same location so we can run the script with
# no path prefix
# WORKDIR /tmp

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install tqdm scikit-learn opencv-python easydict

# CMD defines the command that containers will run when
# created from this image
# CMD ["python", "tftest.py"]
