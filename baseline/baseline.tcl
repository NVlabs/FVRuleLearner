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

# List assertions with counterexamples
set cex_assertions [check_cov -list -status cex]

# Output the failing assertions
puts "Assertions with CEX issues: $cex_assertions"

# If there are any failed properties, output a message
if {[llength $cex_assertions] > 0} {
    puts "Proof failed. Excluding properties with CEX issues from coverage calculation."
}

# Measure coverage, excluding failed assertions by using the -exclude option
check_cov -measure -type all -verbose

# Generate the coverage report, excluding waived, unreachable, deadcode, and CEX items
check_cov -report -exclude {waived unreachable deadcode cex} -model all -type all

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
            # For 'coi' type, use the checker_mode option and exclude failing assertions
            set coverage_data [check_cov -report -model $model -type checker -checker_mode coi -exclude {waived unreachable deadcode cex}]
        } else {
            # For other types, use the regular check_cov command and exclude failing assertions
            set coverage_data [check_cov -report -model $model -type $type -exclude {waived unreachable deadcode cex}]
        }
        # Extract coverage percentage using regexp
        if {[regexp {([0-9.]+)%} $coverage_data match coverage]} {
            puts "$type|$model|$coverage"
        } else {
            puts "$type|$model|N/A"
        }
    }
}
