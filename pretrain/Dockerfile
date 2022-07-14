FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0

RUN pip3 install parse tensorboard filelock

COPY . /torch-segmentation-3d
RUN cd /torch-segmentation-3d && pip3 install .

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY pretrain/src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

RUN ls

ENTRYPOINT ["python3"]