rule collect_defended:
    """"Collect QUIC traces shaped with dummy streams from a dependencies-csv"""
    input:
        "deps-sample.csv"
    output:
        "results/collect/collect_cap.pcapng"
    log:
        "results/collect/collect.log"
    params:
        neqo_path= config['neqo'],
        urls=("https://vanilla.neqo-test.com:7443/" "https://vanilla.neqo-test.com:7443/css/bootstrap.min.css"
             " https://vanilla.neqo-test.com:7443/css/fontAwesome.css" "https://vanilla.neqo-test.com:7443/css/hero-slider.css"
             " https://vanilla.neqo-test.com:7443/css/templatemo-main.css" "https://vanilla.neqo-test.com:7443/css/owl-carousel.css"
             " https://vanilla.neqo-test.com:7443/js/vendor/modernizr-2.8.3-respond-1.4.2.min.js"
             " https://vanilla.neqo-test.com:7443/img/1st-item.jpg" "https://vanilla.neqo-test.com:7443/img/2nd-item.jpg"
             " https://vanilla.neqo-test.com:7443/img/3rd-item.jpg" "https://vanilla.neqo-test.com:7443/img/4th-item.jpg"
             " https://vanilla.neqo-test.com:7443/img/5th-item.jpg" "https://vanilla.neqo-test.com:7443/img/6th-item.jpg"
             " https://vanilla.neqo-test.com:7443/img/1st-tab.jpg" "https://vanilla.neqo-test.com:7443/img/2nd-tab.jpg"
             " https://vanilla.neqo-test.com:7443/img/3rd-tab.jpg" "https://vanilla.neqo-test.com:7443/img/4th-tab.jpg"
             " https://vanilla.neqo-test.com:7443/js/vendor/bootstrap.min.js" "https://vanilla.neqo-test.com:7443/js/plugins.js"
             " https://vanilla.neqo-test.com:7443/js/main.js" "https://vanilla.neqo-test.com:7443/img/1st-section.jpg"
             " https://vanilla.neqo-test.com:7443/img/2nd-section.jpg" "https://vanilla.neqo-test.com:7443/img/3rd-section.jpg"
             " https://vanilla.neqo-test.com:7443/img/4th-section.jpg" "https://vanilla.neqo-test.com:7443/img/5th-section.jpg"
             " https://vanilla.neqo-test.com:7443/fonts/fontawesome-webfont.woff2?v=4.7.0" "https://vanilla.neqo-test.com:7443/img/prev.png"
             " https://vanilla.neqo-test.com:7443/img/next.png" "https://vanilla.neqo-test.com:7443/img/loading.gif"
             " https://vanilla.neqo-test.com:7443/img/close.png" ),
        dummy_urls=("https://vanilla.neqo-test.com:7443/img/2nd-big-item.jpg"
                    " https://vanilla.neqo-test.com:7443/css/bootstrap.min.css"
                    " https://vanilla.neqo-test.com:7443/img/3rd-item.jpg"
                    " https://vanilla.neqo-test.com:7443/img/4th-item.jpg"
                    " https://vanilla.neqo-test.com:7443/img/5th-item.jpg"),
        dummy_ids="",
        dummy_schedule=""
    shell: """\
        docker exec -w {params.neqo_path} -e LD_LIBRARY_PATH={params.neqo_path}/target/debug/build/neqo-crypto-044e50838ff4228a/out/dist/Debug/lib/ \
        -e SSLKEYLOGFILE=out.log -e RUST_LOG= \
        -e CSDEF_NO_SHAPING= neqo-qcd {params.neqo_path}/target/debug/neqo-client  --dummy-urls {params.dummy_urls} {params.urls} 2> {log}
        """