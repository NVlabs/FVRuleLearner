# Analyze property files 
clear -all 

# Initialize coverage for both stimuli models and COI 
check_cov -init -model all -type all -exclude_module {TB_MODULE}

analyze -clear 
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sv 
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva 

# Elaborate design and properties 
elaborate -top ${TOP}

# Define clock and reset signals
clock ${CLK}
reset ${RST}

# Get design information to check general complexity
get_design_info

# Run proof on all assertions with a time limit 
prove -all -time_limit 1m

# Get proof results
set proofs_status [get_status [get_property_list -include {type {assert} disabled {0}}]]

# Output the proof results
puts "proofs: $proofs_status"

# Check if any properties failed (have status 'cex' or 'falsified')
set failed_props [get_property_list -include {type {assert} status {cex falsified}}]

if {[llength $failed_props] > 0} {
    puts "Proof failed. Skipping coverage calculation."
} else {
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

    # Insert Start Marker
    puts "### COVERAGE_REPORT_START ###"

    # List coverage items with status 'undetectable' in CSV format
    set undetectable_coverage [check_cov -list -status undetectable -checker_mode coi]
    puts "### UNDETECTABLE_START ###"
    puts $undetectable_coverage
    puts "### UNDETECTABLE_END ###"

    # List coverage items with status 'unprocessed' in CSV format
    set unprocessed_coverage [check_cov -list -status unprocessed -checker_mode coi]
    puts "### UNPROCESSED_START ###"
    puts $unprocessed_coverage
    puts "### UNPROCESSED_END ###"

    # Insert End Marker
    puts "### COVERAGE_REPORT_END ###"
}
