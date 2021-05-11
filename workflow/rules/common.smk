import pandas as pd
import common


rule common_trace_composition:
    """Extract the composition of a trace in terms of chaff and application data
    as a CSV."""
    output:
        "{dir}/trace-composition.csv"
    input:
        trace="{dir}/trace.pcapng",
        stdout="{dir}/stdout.txt"
    run:
        pd.DataFrame(
            common.pcap.to_quic_lengths(str(input["trace"]), str(input["stdout"]))
        ).to_csv(str(output[0]), header=True, index=False)
