FROM pytorch/pytorch

WORKDIR /app/

RUN apt-get upgrade -y && apt-get update -y

RUN pip install  torch==2.0.1 transformers==4.30.0 librosa==0.10.0 datasets==2.14.6 soundfile==0.12.1 numpy==1.24.3 tqdm==4.65.0 matplotlib==3.10.0 seaborn==0.13.2 

RUN mkdir /app/data/
RUN mkdir /app/code/
