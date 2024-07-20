FROM amazon/aws-lambda-python:3.11

WORKDIR /var/task/lambda-chat

COPY lambda_function.py /var/task/
COPY . ..

RUN /var/lang/bin/python3 -m pip install langchain
RUN /var/lang/bin/python3 -m pip install langchain-community
RUN /var/lang/bin/python3 -m pip install langchain-aws
RUN /var/lang/bin/python3 -m pip install opensearch-py
RUN /var/lang/bin/python3 -m pip install pypdf
RUN /var/lang/bin/python3 -m pip install PyMuPDF
RUN /var/lang/bin/python3 -m pip install pycryptodome
RUN /var/lang/bin/python3 -m pip install python-pptx
RUN /var/lang/bin/python3 -m pip install python-docx
CMD ["lambda_function.lambda_handler"]