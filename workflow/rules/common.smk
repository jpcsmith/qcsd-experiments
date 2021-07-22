def build_neqo_args(exp_config):
    def _builder(wildcards):
        if wildcards["defence"] == "front":
            args = exp_config["front"]
            return [
                "--defence", "front",
                "--defence-packet-size", args["packet_size"],
                "--front-max-client-pkts", args["max_client_packets"],
                "--front-max-server-pkts", args["max_server_packets"],
                "--front-peak-max", args["peak_maximum"],
                "--front-peak-min", args["peak_minimum"],
            ]
        if wildcards["defence"] == "tamaraw":
            args = exp_config["tamaraw"]
            result = [
                "--defence", "tamaraw",
                "--defence-packet-size", args["packet_size"],
                "--tamaraw-rate-in", args["rate_in"],
                "--tamaraw-rate-out", args["rate_out"],
                "--tamaraw-modulo", args["packet_multiple"],
            ]
            if "msd_limit_excess" in args:
              result += ["--msd-limit-excess", args["msd_limit_excess"]]
            return result
        if wildcards["defence"] == "undefended":
            return ["--defence", "none"]
        raise ValueError("Unsupported defence: %r", wildcards["defence"])
    return _builder
