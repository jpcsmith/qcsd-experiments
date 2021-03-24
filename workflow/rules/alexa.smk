"""Download Alexa top-1m list and split it into smaller test lists.

Outputs of this rule are placed in the resources directory, as this rule
is not intended to be run frequently.
"""
from datetime import datetime as dt


rule alexa_1m_today:
    """Download the Alexa top 1m list for today's date."""
    output:
        "resources/alexa-1m-{0}.csv.gz".format(dt.today().strftime('%Y-%m-%d'))
    shell:
        "curl http://s3.amazonaws.com/alexa-static/top-1m.csv.zip"
        " | zcat | gzip --stdout > {output}"


rule alexa_top:
    """Select the top n from the alexa top-1m list."""
    input:
        "resources/alexa-1m-{date}.csv.gz"
    output:
        "resources/alexa-top{topn}-{date}.csv.gz"
    wildcard_constraints:
        date="\d{4}-\d{2}-\d{2}"
    params:
        topn="{topn}"
    shell:
        "set +o pipefail;"
        " zcat {input} | head -n {params.topn} | gzip --stdout > {output}"


rule alexa:
    """Download the Alexa top 1m for today's date and split it into a number of
    smaller lists."""
    input:
        expand([
            "resources/alexa-1m-{date}.csv.gz", "resources/alexa-top10-{date}.csv.gz",
            "resources/alexa-top100-{date}.csv.gz", "resources/alexa-top1000-{date}.csv.gz",
        ], date=dt.today().strftime('%Y-%m-%d'))
