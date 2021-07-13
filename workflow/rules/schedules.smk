wildcard_constraints:
    mbps="[\d\.]+",
    duration="\d+",
    interval="\d+",


rule schedule_constant_rate:
    """Create a schedule with a constant transmission rate for the
    specified duration and interval."""
    output:
        "{output_dir}/constant_{mbps}Mbps_{duration}s_{interval}ms.csv"
    run:
        duration_ms = int(wildcards.duration) * 1000
        interval_ms = int(wildcards.interval)
        # Burst size in bytes per interval_ms
        burst_size = int(float(wildcards.mbps) * 1e3 * interval_ms // 8)

        with open(output[0], mode="w") as outfile:
            for timestamp in range(interval_ms, duration_ms, interval_ms):
                outfile.write(f"{timestamp / 1e3:.3f},{burst_size:d}\n")
                outfile.write(f"{timestamp / 1e3:.3f},{-1 * burst_size:d}\n")
