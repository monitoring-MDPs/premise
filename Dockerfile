FROM lukovdm/stormvogel:premise

RUN apt-get update && apt-get install texlive-latex-recommended texlive-latex-extra parallel -y 

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir /opt/premise
WORKDIR /opt/premise

COPY pyproject.toml ./
COPY poetry.lock ./
COPY README.md ./

# Load existing virtual environment
RUN poetry env use /opt/venv/bin/python

# Install project dependencies
RUN poetry install --without stormpy,sv,carla,dev --no-root

# ENV CARLA_ENV=/opt/carla-venv
# ENV OLD_PATH=$PATH
# RUN pyenv global 3.10
# ENV PATH="$CARLA_ENV/bin:$PATH"

# COPY carla-requirements.txt ./
# RUN pip install --no-cache-dir -r carla-requirements.txt

# ENV PATH=$OLD_PATH

COPY premise .