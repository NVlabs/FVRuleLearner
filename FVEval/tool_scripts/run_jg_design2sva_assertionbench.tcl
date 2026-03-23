# Analyze property files 
clear -all 
# Initialize coverage for both stimuli models and COI 
check_cov -init -model all -type all -exclude_module {TB_MODULE}
analyze -clear 
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sv 
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva 
# Elaborate design and properties 
elaborate -top ${TOP}
# set top [get_inst_top] 
# puts "top: $top" 
# set_reset_max_iterations 1000 
clock ${CLK}
reset ${RST}

# Get design information to check general complexity
get_design_info

# Run proof on all assertions with a time limit 
prove -all
puts "proofs: [get_status [get_property_list -include {type {assert} disabled {0}}]]" 
# Measure coverage for both stimuli models and COI 
check_cov -measure -type all -verbose 
# Coverage reporting script 
# Define the list of coverage models and coverage types
set coverage_models {functional statement toggle expression branch}
set coverage_types {stimuli coi}

# Output header
puts "\nCOVERAGE REPORT"
puts "TYPE|MODEL|COVERAGE"
puts "--------------------"

# Loop over coverage types
foreach type $coverage_types {
    # Loop over coverage models
    foreach model $coverage_models {
        if {$type == "coi"} {
            # For 'coi' type, use the checker_mode option
            set coverage_data [check_cov -report -model $model -type checker -checker_mode coi]
        } else {
            # For other types, use the regular check_cov command
            set coverage_data [check_cov -report -model $model -type $type]
        }
        # Extract coverage percentage using regexp
        if {[regexp {([0-9.]+)%} $coverage_data match coverage]} {
            puts "$type|$model|$coverage"
        } else {
            puts "$type|$model|N/A"
        }
    }
}
