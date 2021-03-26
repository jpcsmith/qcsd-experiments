# Anoa consists of two components:
# 1. Send packets at some packet rate until data is done.
# 2. Pad to cover total transmission size.
# The main logic decides how to send the next packet.
# Resultant anonymity is measured in ambiguity sizes.
# Resultant overhead is in size and time.
# Maximizing anonymity while minimizing overhead is what we want.

import math
import random
from time import strftime
import argparse
import logging
import numpy as np
import pandas as pd
import sys
import os


logger = logging.getLogger('tamaraw')

'''params'''
DATASIZE = 750
PadL = 100

# Logging format
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"


tardist = [[], []]
defpackets = []


def fsign(num):
    if num > 0:
        return 0
    else:
        return 1


def rsign(num):
    if num == 0:
        return 1
    else:
        return abs(num)/num


def AnoaTime(parameters):
    direction = parameters[0]  # 0 out, 1 in
    method = parameters[1]
    if (method == 0):
        if direction == 0:
            return 0.02
        if direction == 1:
            return 0.006


def AnoaPad(list1, list2, padL, method):
    lengths = [0, 0]
    times = [0, 0]
    for x in list1:
        if (x[1] > 0):
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)

    paddings = []

    for j in range(0, 2):
        curtime = times[j]
        # 1/2 1, 1/4 2, 1/8 3, ... #check this
        topad = -int(math.log(random.uniform(0.00001, 1), 2) - 1)
        if (method == 0):
            if padL == 0:
                topad = 0
            else:
                topad = (lengths[j]//padL + topad) * padL

        logger.info("Need to pad %d packets." % topad)
        while (lengths[j] < topad):
            curtime += AnoaTime([j, 0])
            if j == 0:
                paddings.append([curtime, DATASIZE])
            else:
                paddings.append([curtime, -DATASIZE])
            lengths[j] += 1
    paddings = sorted(paddings, key=lambda x: x[0])
    list2.extend(paddings)


def Anoa(list1, list2, parameters):  # inputpacket, outputpacket, parameters
    # Does NOT do padding, because ambiguity set analysis.
    # list1 WILL be modified! if necessary rewrite to tempify list1.
    starttime = list1[0][0]
    times = [starttime, starttime]  # lastpostime, lastnegtime
    curtime = starttime
    lengths = [0, 0]
    datasize = DATASIZE
    method = 0
    if (method == 0):
        parameters[0] = "Constant packet rate: " + str(AnoaTime([0, 0])) + ", " + str(AnoaTime([1, 0])) + ". "
        parameters[0] += "Data size: " + str(datasize) + ". "
    if (method == 1):
        parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
        parameters[0] += "Tolerance: 2x."
    listind = 0  # marks the next packet to send
    while (listind < len(list1)):
        #decide which packet to send
        if times[0] + AnoaTime([0, method, times[0]-starttime]) < times[1] + AnoaTime([1, method, times[1]-starttime]):
            cursign = 0
        else:
            cursign = 1
        times[cursign] += AnoaTime([cursign, method, times[cursign]-starttime])
        curtime = times[cursign]

        tosend = datasize
        while (list1[listind][0] <= curtime and fsign(list1[listind][1]) == cursign and tosend > 0):
            if (tosend >= abs(list1[listind][1])):
                tosend -= abs(list1[listind][1])
                listind += 1
            else:
                list1[listind][1] = (abs(list1[listind][1]) - tosend) * rsign(list1[listind][1])
                tosend = 0
            if (listind >= len(list1)):
                break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        lengths[cursign] += 1


def parse_arguments():
    # Read configuration file
    # conf_parser = configparser.RawConfigParser()
    # conf_parser.read(ct.CONFIG_FILE)

    parser = argparse.ArgumentParser(description='It simulates tamaraw on a set of web traffic traces.')

    # parser.add_argument('traces_path',
    #                     metavar='<traces path>',
    #                     help='Path to the directory with the traffic traces to be simulated.')

    parser.add_argument('output_path',
                        metavar='<output path>',
                        help='Path to the output file containing tamaraw defended trace.')

    parser.add_argument('baseline_path',
                        metavar='<baseline path>',
                        help='Path to the baseline csv.')

    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')

    args = parser.parse_args()
    config_logger(args)

    return args


def config_logger(args):
    # Set file
    log_file = sys.stdout
    if args.log != 'stdout':
        log_file = open(args.log, 'w')
    ch = logging.StreamHandler(log_file)

    # Set logging format
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)

    # Set level format
    logger.setLevel(logging.INFO)


def load_trace(filename):
    data = pd.read_csv(filename)
    assert data.loc[0, "packet_number"] == 0
    data["timestamp"] = (data["timestamp"] - data.loc[0, "timestamp"])
    data = data.rename(columns={"timestamp": "time",
                                "packet_length": "length"})
    _out = data["is_outgoing"] == True
    _in = data["is_outgoing"] == False
    data.loc[_in, "length"] = -data[_in]["length"]
    data = data[["time", "length"]]
    return data


def create_target(fname, output):
    logger.info('Simulating %s...' % fname)
    baseline = load_trace(fname)
    baseline = baseline.values
    print("baseline len: ", len(baseline))

    list2 = []
    parameters = [""]

    Anoa(baseline, list2, parameters)
    list2 = sorted(list2, key = lambda list2: list2[0])
    anoad.append(list2)

    print("list2 len: ", len(list2))

    list3 = []

    AnoaPad(list2, list3, PadL, 0)

    print("list3 len: ", len(list3))

    # fout = open(output, "w")
    # for x in list3:
    #     fout.write(str(x[0]) + "," + str(x[1]) + "\n")
    # fout.close()

    # sizes:
    old = sum([abs(p[1]) for p in baseline])
    mid = sum([abs(p[1]) for p in list2])
    new = sum([abs(p[1]) for p in list3])
    logger.info("old size:%d,mid size:%d, new size:%d" % (old, mid, new))

    return list3


if __name__ == '__main__':
    args = parse_arguments()
    logger.info("Arguments: %s" % (args))
    foldout = args.output_path

    packets = []
    desc = ""
    anoad = []
    anoadpad = []
    latencies = []
    sizes = []
    bandwidths = []

    tot_new_size = 0.0
    tot_new_latency = 0.0
    tot_old_size = 0.0
    tot_old_latency = 0.0

    target = create_target(args.baseline_path, args.output_path)

    pd.DataFrame(target).to_csv(args.output_path, index=False, header=False)
