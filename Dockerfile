FROM python:3.9.13-slim-buster

ENV TZ=Asia/Shanghai DEBIAN_FRONTEND=noninteractive

RUN sed -i s/deb.debian.org/mirrors.ustc.edu.cn/g /etc/apt/sources.list

RUN apt update \
    && apt install -y tzdata fonts-noto-cjk \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY ./ /app/

EXPOSE 8080

CMD python run.py
