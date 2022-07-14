#!/bin/sh

docker build -t eidos-service.di.unito.it/barbano/torch-segmentation-3d:pretrain . -f pretrain/Dockerfile.pretrain
docker push eidos-service.di.unito.it/barbano/torch-segmentation-3d:pretrain

docker build -t eidos-service.di.unito.it/barbano/torch-segmentation-3d:sweep . -f pretrain/Dockerfile.sweeo
docker push eidos-service.di.unito.it/barbano/torch-segmentation-3d:sweep
