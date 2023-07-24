from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scapy.all import *

ip_address = {'yahoo': '74.6.231.20', 'barreausol': '194.59.214.115', 'youtube': '172.217.14.206'}
ip_name = 'youtube'
packets = rdpcap("./wireshark/" + ip_name + ".pcap")

# packets = sniff(offline='./wireshark/yahoo.pcap')
filtered_sessions = []
i = 0
for key, sessions in packets.sessions().items():
    if key.split()[0] != 'TCP' or key.split()[1].split(':')[0] != ip_address[ip_name]:
        continue
    i += 1
    temp_session = []
    for packet in sessions:
        temp_session.append(packet)
    filtered_sessions.append(temp_session)


packet_lengths = []
for sessions in filtered_sessions:
    temp_lengths = []
    for packet in sessions:
        temp_lengths.append(packet[IP].len)
    packet_lengths.append(temp_lengths)

# print(len(packet_lengths[0]))

# plt.plot(range(0, 992), packet_lengths[0][0:992], marker='', color='red', linewidth=2)
# plt.plot(range(0, 100), packet_lengths[1][0:100], marker='', color='blue', linewidth=2)
# plt.plot(range(0, 14), packet_lengths[2], marker='', color='olive', linewidth=2)
# plt.plot(range(0, 16), packet_lengths[3], marker='', color='black', linewidth=2)


plt.hist(packet_lengths[0], bins=100)
plt.ylabel('Packets')
plt.xlabel('Number')

plt.show()