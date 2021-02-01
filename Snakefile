configfile: "config/config.yaml"

include: "rules/profile-domains.smk"
include: "rules/determine-url-deps.smk"
include: "rules/collect.smk"
include: "rules/pearson.smk"
include: "rules/icrc-demo.smk"
