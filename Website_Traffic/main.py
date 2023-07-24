from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scapy.all import *
import seaborn as sns
import sys
import statistics
import cv2
import os


pad_numpy = lambda a, i: a[0: i] if a.shape[0] > i else np.hstack((a, np.zeros(i - a.shape[0])))
pcapfile = rdpcap("./dataset/darpa.pcap")

filtered_sessions = []
protocol_separated_sessions = {}
interested_protocols = ['IP', 'ICMP', 'TCP', 'UDP']

for key, sessions in pcapfile.sessions().items():
    if key.split()[0] not in protocol_separated_sessions:
        protocol_separated_sessions[key.split()[0]] = []

    temp_session = []
    for packet in sessions:
        temp_session.append(packet)
    protocol_separated_sessions[key.split()[0]].append(temp_session)

protocol_separated_sessions = dict((key, value) for key, value in protocol_separated_sessions.items() if key in interested_protocols)

# protocol_separated_sessions_decimals = {'IP': [], 'ICMP': [], 'TCP': [], 'UDP': []}
processed_protocol_separated_sessions = {'IP': [], 'ICMP': [], 'TCP': [], 'UDP': []}

# for key, protocol in protocol_separated_sessions.items():
#     if key == 'ICMP':
    #     packets_decimal = []
    #     for session in protocol:
    #         for packet in session:
    #             temp_packet_decimal = []
    #             packet_bytes = (chexdump(packet, dump=True).split(', '))
    #             for byte in packet_bytes:
    #                 temp_packet_decimal.append(int(byte, 0))
    #             if len(temp_packet_decimal) == 98:
    #                 packets_decimal.append(np.array(temp_packet_decimal, dtype=int))
    #
    #     stacked_packet_decimals = np.stack(packets_decimal, axis=0)
    #     protocol_separated_sessions_decimal[key].append(stacked_packet_decimals)
    # if key == 'ICMP':
    #     continue
    #     for session in protocol:
    #         temp_session = []
    #         for packet in session:
    #             temp_packet_features = ['ip_id', 'ip_flags', 'ip_frag', 'ip_ttl', 'icmp_type', 'icmp_code', 'icmp_id',
    #                                     'icmp_seq', 'icmp_unused']
    #             # temp_packet = []
    #             temp_packet = np.zeros((8, 1))
    #             i = 0
    #             for feature in temp_packet_features:
    #                 if feature == 'ip_flags':
    #                     temp_packet[i, 0] = str(getattr(packet[feature.split('_')[0].upper()], feature.split('_')[1]))
    #                     # temp_packet.append(str(getattr(packet[feature.split('_')[0].upper()], feature.split('_')[1])))
    #                 elif feature == 'icmp_unused':
    #                     temp_packet[i, 0] = getattr(packet[feature.split('_')[0].upper()], feature.split('_')[1]).decode('utf-8')
    #                     # temp_packet.append(getattr(packet[feature.split('_')[0].upper()], feature.split('_')[1]).decode('utf-8'))
    #                 else:
    #                     temp_packet[i, 0] = getattr(packet[feature.split('_')[0].upper()], feature.split('_')[1])
    #                     # temp_packet.append(getattr(packet[feature.split('_')[0].upper()], feature.split('_')[1]))
    #                 i += 1
    #             temp_session.append(temp_packet)
    #         processed_protocol_separated_sessions[key].append(temp_session)
    # elif key == 'TCP':
    #     for session in protocol:
    #         for packet in session:
    #             if packet[IP].src != '207.46.130.139':
    #                 break
    #             temp_packet_decimal = []
    #             packet_bytes = (chexdump(packet, dump=True).split(', '))
    #             for byte in packet_bytes:
    #                 temp_packet_decimal.append(int(byte, 0))
    #             if len(temp_packet_decimal) == 60:
    #                 print(packet.show())
    #     break
    # else:
    #     continue


packet_session_stats_array = {'IP': [[]], 'ICMP': {'std': [], 'var': []}, 'TCP': [[]], 'UDP': [[]]}
# for session in protocol_separated_sessions_decimal['ICMP']:
#     packet_session_stats_array['ICMP']['std'] = np.std(session, axis=0)
#     break
#

session_idx = 0
for session in protocol_separated_sessions['TCP']:
    session_idx += 1
    packet_idx = 0
    for packet in session:
        packet_idx += 1
        if packet[IP].src != '207.46.130.139':
            break

        # data_bytes = packet[Raw].__bytes__()
        # data_bytes = packet.__bytes__()
        packet[TCP].remove_payload()
        data_bytes = packet[IP].__bytes__() + packet[TCP].__bytes__()
        data_bytes = bytearray(data_bytes)
        flatNumpyArray = np.array(data_bytes)
        flatNumpyArray = pad_numpy(flatNumpyArray, 64)


        grayImage = flatNumpyArray.reshape(8, 8)
        # cv2.imwrite('img.png', grayImage)
        cv2.imwrite('small_packet_images/packet' + str(packet[IP].id) + '.png', grayImage)

sys.exit(1)


# packet_lengths = []
# for sessions in protocol_separated_sessions:
#     temp_lengths = []
#     for packet in sessions:
#         temp_lengths.append(packet[IP].len)
#     packet_lengths.append(temp_lengths)

# print(len(packet_lengths[0]))
# print(len(packet_lengths[1]))
# print(len(packet_lengths[2]))
# print(len(packet_lengths[3]))

# colors = ['red', 'blue', 'olive', 'black']
# for x in range(0, 4):
#     plt.plot(range(0, len(packet_lengths[x])), packet_lengths[x], marker='', color=colors[x], linewidth=2)

# i = 10
# for packet_length in packet_lengths:
#     plt.hist(packet_length, bins=100)
#     plt.ylabel('Packets')
#     plt.xlabel('Number')
#     plt.show()
#     i -= 1
#     if i == 0:
#         break

# plt.hist(packet_lengths[0:10], bins=10)
# plt.ylabel('Packets')
# plt.xlabel('Number')
# plt.show()

# fig, ax = plt.subplots(figsize=(30, 10))
# sns.heatmap(np.reshape(packet_session_stats_array['ICMP']['std'], (1, -1)),  square=True, annot=True, cmap='Greys', ax=ax)
# bx = sns.heatmap(np.reshape(packet_session_stats_array['ICMP']['var'], (1, -1)))
# plt.imshow(np.reshape(packet_session_stats_array['ICMP']['std'], (1, -1)), cmap='hot', interpolation='nearest')
# plt.show()