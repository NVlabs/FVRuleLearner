# Clear previous data
clear -all

# Analyze design and assertion files
analyze -clear
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sv
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva

# Elaborate design
elaborate
set top [get_inst_top]
puts "top: $top"

# Define clock and reset signals
clock clk
reset -expression (${top}_tb_inst.tb_reset)

# Run proof on all assertions with a time limit
prove -all -time_limit 1m

# Output the proof results
puts "proofs: [get_status [get_property_list -include {type {assert} disabled {0}}]]"

puts "coverage: [get_status [get_property_list -include {type {cover} disabled {0}}]]"