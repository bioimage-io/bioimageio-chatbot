FROM continuumio/miniconda3:latest
RUN pip install bioimageio-chatbot

EXPOSE 9000