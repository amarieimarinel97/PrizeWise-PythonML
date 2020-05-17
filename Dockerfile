FROM tensorflow/tensorflow:latest-gpu
COPY . .


ENV PYTHONPATH=/Regression
RUN pip install matplotlib 
RUN pip install pandas 
RUN pip install cherrypy 
RUN pip install scikit-learn 
RUN pip install tensorflow_datasets 
RUN pip install tensorflow_hub
EXPOSE 8081
WORKDIR /Regression/service
CMD python service.py

#docker build -t python:tag .
#docker run -it --network diploma-proj-net -p 8081:8081 --name diploma-proj-python --rm python:tag
