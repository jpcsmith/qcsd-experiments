rule test:
  input:
    url_dep="results/determine-url-deps/dependencies/000.json"
  output:
    out_dir=directory("test")
  params:
    seed=42
  script:
    "../scripts/run_front_experiment.py"
