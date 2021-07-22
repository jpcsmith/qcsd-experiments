"""Create a latex table of the overhead plots."""
import textwrap
import pandas as pd


def main(front: str, tamaraw: str, output: str):
    """Create a latex table of the overhead plots."""
    with open(output, mode="w") as table:
        table.write(textwrap.dedent(r"""
            \begin{tabular}{@{}lrr@{}}
            \toprule
            & FRONT & Tamaraw \\
            \midrule
            """))

        data = pd.concat([pd.read_csv(front), pd.read_csv(tamaraw)],
                         keys=["front", "tamaraw"], names=["defence"])
        data = (data.groupby(["defence", "overhead", "setting"])["value"]
                    .describe())
        data = data.rename(index={
            'simulated': 'Simulated', "collected": "Defended",
            "bandwidth": "Bandwidth", "latency": "Latency",
        })

        for overhead in ["Bandwidth", "Latency"]:
            table.write(r"\textbf{%s} \\" % overhead)
            table.write("\n")
            for setting in ["Defended", "Simulated"]:
                table.write(r"\quad %s" % setting)
                if overhead == "Latency" and setting == "Simulated":
                    table.write(r" & $0.00$ & $-$ \\" + "\n")
                    continue
                for defence in ["front", "tamaraw"]:
                    median = data["50%"][defence][overhead][setting]
                    iqr1 = data["25%"][defence][overhead][setting]
                    iqr3 = data["75%"][defence][overhead][setting]
                    table.write(f" &${median:.2f}$ (${iqr1:.2f} - {iqr3:.2f}$)")
                table.write(r" \\" + "\n")
        table.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(front=snakemake.input["front"], tamaraw=snakemake.input["tamaraw"],
         output=snakemake.output[0])
