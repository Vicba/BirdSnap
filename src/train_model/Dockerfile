FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["bash", "train.sh"]