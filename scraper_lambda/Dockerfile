FROM public.ecr.aws/lambda/python:3.12

RUN yum install -y chromium chromium-headless
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scraper.py .

CMD ["scraper.lambda_handler"]
