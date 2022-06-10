FROM jinaai/jina:3-py37-perf

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install --default-timeout=1000 --compile -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

EXPOSE 8510

ENTRYPOINT ["streamlit"]
CMD ["run", "frontend.py"]
