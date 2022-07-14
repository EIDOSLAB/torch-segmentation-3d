#!/bin/sh

docker build -t eidos-service.di.unito.it/barbano/torch-segmentation-3d:pretrain . -f Dockerfile
docker push eidos-service.di.unito.it/barbano/torch-segmentation-3d:pretrain
