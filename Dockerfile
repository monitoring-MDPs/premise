FROM sjunges/stormpy:1.6.3

RUN apt-get update && apt-get install texlive-latex-recommended texlive-latex-extra -y 

RUN mkdir /opt/premise
WORKDIR /opt/premise

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup.py install



