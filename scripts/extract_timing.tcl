read_liberty /path/to/library.lib
read_verilog data/design.v
link_design top_module
read_sdc data/constraints.sdc
report_timing -format json > data/timing_report.json

