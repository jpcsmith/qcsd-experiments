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

            settings = ["Defended", "Simulated"]
            if overhead == "Bandwidth":
                settings.append("simulated-alt")

            for setting in settings:
                table.write(r"\quad %s" % setting)
                for defence in ["front", "tamaraw"]:
                    median = data["50%"][defence][overhead][setting]
                    iqr1 = data["25%"][defence][overhead][setting]
                    iqr3 = data["75%"][defence][overhead][setting]
                    table.write(
                        f" &${median:.2f}$ (${iqr1:.2f}\\text{{--}}{iqr3:.2f}$)"
                    )
                table.write(r" \\" + "\n")
        table.write("\\bottomrule\n\\end{tabular}\n")
        with pd.option_context('display.max_colwidth', None):
            table.write(textwrap.indent(
                data.drop(
                    columns=["count", "mean", "std", "min", "max"]).to_csv(),
                "% "))


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(front=snakemake.input["front"], tamaraw=snakemake.input["tamaraw"],
         output=snakemake.output[0])
