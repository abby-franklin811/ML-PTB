FROM python:3

RUN pip install sklearn
RUN pip install pandas

COPY modelTest.py modelTest.py
COPY basic_rf_preterm.save basic_rf_preterm.save

CMD [ "python3", "./modelTest.py"]
