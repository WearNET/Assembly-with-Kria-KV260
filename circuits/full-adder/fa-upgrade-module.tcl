set bd_name "full_adder_design"

set bd_file [get_files -quiet "${bd_name}.bd"]

if {[llength $bd_file] != 1} {
    error "Could not uniquely find ${bd_name}.bd"
}

open_bd_design $bd_file

# Make Vivado rescan the modified RTL
update_compile_order -fileset sources_1

update_module_reference [get_ips  full_adder_design_full_adder_stub_0_0]

# Finish bd design
validate_bd_design
save_bd_design
generate_target all $bd_file

puts "Regenerated block design: $bd_name"

update_compile_order -fileset sources_1
close_bd_design [get_bd_designs full_adder_design]
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

set project_dir [get_property DIRECTORY [current_project]]
set project_name [get_property NAME [current_project]]
set xsa_file [file join $project_dir "${project_name}_wrapper.xsa"]

write_hw_platform -fixed -include_bit -force -file $xsa_file