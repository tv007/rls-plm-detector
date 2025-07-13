from pyedflib import highlevel

signals, signal_headers, header = highlevel.read_edf("data/Gangwar_Kusum_(1).edf")
print("Header:", header)
print("Signal headers:", signal_headers)
print("First 10 samples of first channel:", signals[0][:10]) 